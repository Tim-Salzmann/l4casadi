import casadi as cs
import numpy as np
import torch
import l4casadi as l4c
from acados_template import AcadosOcpSolver, AcadosOcp, AcadosModel
import time

COST = 'LINEAR_LS'  # NONLINEAR_LS


class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.input_layer = torch.nn.Linear(2, 512)

        hidden_layers = []
        for i in range(2):
            hidden_layers.append(torch.nn.Linear(512, 512))

        self.hidden_layer = torch.nn.ModuleList(hidden_layers)
        self.out_layer = torch.nn.Linear(512, 1)

        # Model is not trained -- setting output to zero
        with torch.no_grad():
            self.out_layer.bias.fill_(0.)
            self.out_layer.weight.fill_(0.)

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layer:
            x = torch.tanh(layer(x))
        x = self.out_layer(x)
        return x


class DoubleIntegratorWithLearnedDynamics:
    def __init__(self, learned_dyn):
        self.learned_dyn = learned_dyn

    def model(self):
        s = cs.MX.sym('s', 1)
        s_dot = cs.MX.sym('s_dot', 1)
        s_dot_dot = cs.MX.sym('s_dot_dot', 1)
        u = cs.MX.sym('u', 1)
        x = cs.vertcat(s, s_dot)
        x_dot = cs.vertcat(s_dot, s_dot_dot)

        res_model = self.learned_dyn(x.T).T

        f_expl = cs.vertcat(
            s_dot,
            u
        ) + res_model

        x_start = np.zeros((2, ))

        # store to struct
        model = cs.types.SimpleNamespace()
        model.x = x
        model.xdot = x_dot
        model.u = u
        model.z = cs.vertcat([])
        model.p = cs.vertcat([])
        model.f_expl = f_expl
        model.x_start = x_start
        model.constraints = cs.vertcat([])
        model.name = "wr"

        return model


class MPC:
    def __init__(self, model, N, external_shared_lib_dir, external_shared_lib_name):
        self.N = N
        self.model = model
        self.external_shared_lib_dir = external_shared_lib_dir
        self.external_shared_lib_name = external_shared_lib_name

    @property
    def solver(self):
        return AcadosOcpSolver(self.ocp())

    def ocp(self):
        model = self.model

        t_horizon = 1.
        N = self.N

        # Get model
        model_ac = self.acados_model(model=model)
        model_ac.p = model.p

        # Dimensions
        nx = 2
        nu = 1
        ny = 1

        # Create OCP object to formulate the optimization
        ocp = AcadosOcp()
        ocp.model = model_ac
        ocp.dims.N = N
        ocp.dims.nx = nx
        ocp.dims.nu = nu
        ocp.dims.ny = ny
        ocp.solver_options.tf = t_horizon

        if COST == 'LINEAR_LS':
            # Initialize cost function
            ocp.cost.cost_type = 'LINEAR_LS'
            ocp.cost.cost_type_e = 'LINEAR_LS'

            ocp.cost.W = np.array([[1.]])

            ocp.cost.Vx = np.zeros((ny, nx))
            ocp.cost.Vx[0, 0] = 1.
            ocp.cost.Vu = np.zeros((ny, nu))
            ocp.cost.Vz = np.array([[]])
            ocp.cost.Vx_e = np.zeros((ny, nx))

            l4c_y_expr = None
        else:
            ocp.cost.cost_type = 'NONLINEAR_LS'
            ocp.cost.cost_type_e = 'NONLINEAR_LS'

            x = ocp.model.x

            ocp.cost.W = np.array([[1.]])

            # Trivial PyTorch index 0
            l4c_y_expr = l4c.L4CasADi(lambda x: x[0], name='y_expr')

            ocp.model.cost_y_expr = l4c_y_expr(x)
            ocp.model.cost_y_expr_e = x[0]

        ocp.cost.W_e = np.array([[0.]])
        ocp.cost.yref_e = np.array([0.])

        # Initial reference trajectory (will be overwritten)
        ocp.cost.yref = np.zeros(1)

        # Initial state (will be overwritten)
        ocp.constraints.x0 = model.x_start

        # Set constraints
        a_max = 10
        ocp.constraints.lbu = np.array([-a_max])
        ocp.constraints.ubu = np.array([a_max])
        ocp.constraints.idxbu = np.array([0])

        # Solver options
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.model_external_shared_lib_dir = self.external_shared_lib_dir
        if COST == 'LINEAR_LS':
            ocp.solver_options.model_external_shared_lib_name = self.external_shared_lib_name
        else:
            ocp.solver_options.model_external_shared_lib_name = self.external_shared_lib_name + ' -l' + l4c_y_expr.name

        return ocp

    def acados_model(self, model):
        model_ac = AcadosModel()
        model_ac.f_impl_expr = model.xdot - model.f_expl
        model_ac.f_expl_expr = model.f_expl
        model_ac.x = model.x
        model_ac.xdot = model.xdot
        model_ac.u = model.u
        model_ac.name = model.name
        return model_ac


def run():
    N = 10
    learned_dyn_model = l4c.L4CasADi(MultiLayerPerceptron(), name='learned_dyn')

    model = DoubleIntegratorWithLearnedDynamics(learned_dyn_model)
    solver = MPC(model=model.model(), N=N,
                 external_shared_lib_dir=learned_dyn_model.shared_lib_dir,
                 external_shared_lib_name=learned_dyn_model.name).solver

    x = []
    x_ref = []
    ts = 1. / N
    xt = np.array([1., 0.])
    opt_times = []

    for i in range(50):
        now = time.time()
        t = np.linspace(i * ts, i * ts + 1., 10)
        yref = np.sin(0.5 * t + np.pi / 2)
        x_ref.append(yref[0])
        for t, ref in enumerate(yref):
            solver.set(t, "yref", ref)
        solver.set(0, "lbx", xt)
        solver.set(0, "ubx", xt)
        solver.solve()
        xt = solver.get(1, "x")
        x.append(xt)

        x_l = []
        for i in range(N):
            x_l.append(solver.get(i, "x"))

        elapsed = time.time() - now
        opt_times.append(elapsed)

    print(f'Mean iteration time: {1000*np.mean(opt_times):.1f}ms -- {1/np.mean(opt_times):.0f}Hz)')


if __name__ == '__main__':
    run()