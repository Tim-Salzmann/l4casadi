import casadi as cs
import torch.nn

import l4casadi as l4c

# Declare variables
x = cs.MX.sym("x", 2)


# Form the NLP
class PyTorchObjectiveModel(torch.nn.Module):
    def forward(self, input):
        return torch.square(input[0]) + torch.square(input[1])[..., None]


f = PyTorchObjectiveModel()  # objective
f = l4c.L4CasADi(f, name='f')(x)


class PyTorchConstraintModel(torch.nn.Module):
    def forward(self, input):
        return (input[0] + input[1] - 10)[..., None]


g = PyTorchConstraintModel()  # constraint
g = l4c.L4CasADi(g, name='g')(x)

nlp = {'x': x, 'f': f, 'g': g}


# Pick an NLP solver
MySolver = "ipopt"
# MySolver = "worhp"
# MySolver = "sqpmethod"

# Solver options
opts = {}
if MySolver == "sqpmethod":
    opts["qpsol"] = "qpoases"
    opts["qpsol_options"] = {"printLevel": "none"}  # type: ignore[assignment]

# Allocate a solver
solver = cs.nlpsol("solver", MySolver, nlp, opts)

# Solve the NLP
sol = solver(lbg=0)

# Print solution
print("-----")
print("objective at solution = ", sol["f"])
print("primal solution = ", sol["x"])
print("dual solution (x) = ", sol["lam_x"])
print("dual solution (g) = ", sol["lam_g"])
