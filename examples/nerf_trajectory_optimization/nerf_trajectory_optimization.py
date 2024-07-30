import os

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
import torch

import l4casadi as l4c
from density_nerf import DensityNeRF

CASE = 1

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def polynomial(n, n_eval):
    """Generates a symbolic function for a polynomial of degree n-1"""

    # Polynomial symbolic function
    coeffs = cs.MX.sym("coeffs", n, 3)
    xi = cs.MX.sym("xi")
    p = cs.MX.zeros(1, 3)
    for k in range(n):
        p += coeffs[k, :] * xi**k

    v = cs.jacobian(p, xi).T
    a = cs.jacobian(v, xi).T
    j = cs.jacobian(a, xi).T
    s = cs.jacobian(j, xi).T

    f = cs.Function(
        "f_poly",
        [coeffs, xi],
        [p, v, a, j, s],
        ["coeffs", "xi"],
        ["p", "v", "a", "j", "s"],
    )

    # evaluation function
    p_eval = cs.MX.zeros(n_eval, 3)
    v_eval = cs.MX.zeros(n_eval, 3)
    a_eval = cs.MX.zeros(n_eval, 3)
    j_eval = cs.MX.zeros(n_eval, 3)
    s_eval = cs.MX.zeros(n_eval, 3)
    xi_eval = np.linspace(0, 1, n_eval)
    for k in range(n_eval):
        p_eval[k, :] = f(coeffs=coeffs, xi=xi_eval[k])["p"]
        v_eval[k, :] = f(coeffs=coeffs, xi=xi_eval[k])["v"]
        a_eval[k, :] = f(coeffs=coeffs, xi=xi_eval[k])["a"]
        j_eval[k, :] = f(coeffs=coeffs, xi=xi_eval[k])["j"]
        s_eval[k, :] = f(coeffs=coeffs, xi=xi_eval[k])["s"]

    f_eval = cs.Function(
        "f_eval",
        [coeffs],
        [p_eval, v_eval, a_eval, j_eval, s_eval],
        ["coeffs"],
        ["p", "v", "a", "j", "s"],
    )

    return f, f_eval


def trajectory_generator_solver(n, n_eval, L, warmup, threshold):
    # Decision variables and parameters
    f_poly, f_eval = polynomial(n, n_eval)
    x = cs.MX.sym("x", n, 2)
    X = cs.horzcat(cs.MX.zeros(n), x)
    params = cs.MX.sym("params", n_eval, 3)
    x_init = params[0, :]
    x_end = params[-1, :]

    # Define NLP
    f = 0
    g = []
    lbg = []
    ubg = []

    for k in range(n_eval):
        poly = f_poly(coeffs=X, xi=k / (n_eval - 1))
        pk = poly["p"]
        sk = poly["s"]

        if warmup:
            f += cs.sum2((pk - params[k, :]) ** 2)
        else:
            # Optimize for minimum Snap.
            f += cs.sum2(sk**2)

            # While having a maximum density (1.) of the NeRF as constraint.
            lk = L(pk)
            g = cs.horzcat(g, lk)
            lbg = cs.horzcat(lbg, cs.DM([-10e32]).T)
            ubg = cs.horzcat(ubg, cs.DM([threshold]).T)

        # Spatial bounds
        g = cs.horzcat(g, pk[1:])
        lbg = cs.horzcat(lbg, cs.DM([-1, -0.3]).T)
        ubg = cs.horzcat(ubg, cs.DM([1.2, 1.0]).T)

    # Initial and final states
    eps = 0
    for key, init, end in zip(
        ["p"],
        [x_init],
        [x_end],
    ):
        g = cs.horzcat(g, f_poly(coeffs=X, xi=0)[key] - init)
        lbg = cs.horzcat(lbg, -cs.DM([eps, eps, eps]).T)
        ubg = cs.horzcat(ubg, cs.DM([eps, eps, eps]).T)

        g = cs.horzcat(g, f_poly(coeffs=X, xi=1)[key] - end)
        lbg = cs.horzcat(lbg, -cs.DM([eps, eps, eps]).T)
        ubg = cs.horzcat(ubg, cs.DM([eps, eps, eps]).T)

    # Generate solver
    x_nlp = cs.reshape(x, n * 2, 1)
    p_nlp = cs.reshape(params, n_eval * 3, 1)
    nlp_dict = {
        "x": x_nlp,
        "f": f,
        "g": g,
        "p": p_nlp,
    }

    if warmup:
        nlp_opts = {
            "ipopt.linear_solver": "mumps",
            "ipopt.sb": "yes",
            "ipopt.max_iter": 100,
            "ipopt.print_level": 5,
            "print_time": False,
        }
    else:
        nlp_opts = {
            # High barrier parameter to adhere to warmstart.
            "ipopt.mu_init": 1e-4,
            "ipopt.barrier_tol_factor": 1e6,

            "ipopt.linear_solver": "mumps",
            "ipopt.sb": "yes",
            "ipopt.max_iter": 100,
            "ipopt.print_level": 5,
            "print_time": False,
        }

    nlp_solver = cs.nlpsol("nerf_trajectory_optimizer", "ipopt", nlp_dict, nlp_opts)

    solver = {"solver": nlp_solver, "lbg": lbg, "ubg": ubg}

    return solver


def main():
    n = 9
    n_eval = 150
    optimization_threshold = 1.
    viz_threshold = 10.

    if CASE == 1:  # case 1
        p_start = np.array([0.0, -0.8, -0.2])
        p_goal = np.array([-0.0, 1.2, 0.8])
    elif CASE == 2:  # case 2
        p_start = np.array([0.0, -0.8, -0.2])
        p_goal = np.array([-0.0, 1.2, -0.2])
    elif CASE == 3:  # case 3
        p_start = np.array([0.0, -1, 1])
        p_goal = np.array([-0.0, 1.2, -0.2])
    else:
        raise ValueError("Invalid case.")

    # --------------------------------- Load NERF -------------------------------- #
    model = DensityNeRF()
    model_path = os.path.join(os.path.dirname(__file__), "nerf_model.tar")
    model.load_state_dict(
        torch.load(model_path, map_location="cpu")["network_fn_state_dict"],
        strict=False,
    )
    # -------------------------- Create L4CasADi Module -------------------------- #
    l4c_nerf = l4c.L4CasADi(model, scripting=False)

    # ---------------------------------------------------------------------------- #
    #                                   NLP warmup                                   #
    # ---------------------------------------------------------------------------- #

    # --------------------------- Piecewise linear path -------------------------- #

    if CASE == 1:
        points = np.array(
            [[0, -0.8, -0.2], [0, -0.5, 0.4], [0, 0, 0.8], [0, 0.75, 0.3], [0, 1.2, 0.8]]
        )
    elif CASE == 2:
        points = np.array(
            [[0, -0.8, -0.2], [0, -0.5, 0.4], [0, 0, 0.8], [0, 0.75, 0.4], [0, 1.2, -0.2]]
        )
    elif CASE == 3:
        points = np.array(
            [[0.0, -1, 1], [0, -0.85, 0.4], [0, 0, 0.7], [0, 0.75, 0.45], [0, 1.2, -0.2]]
        )
    else:
        raise ValueError("Invalid case")

    dists = np.linalg.norm(np.diff(points, axis=0), axis=1)
    n_eval_points = np.squeeze(dists / np.sum(dists) * n_eval).astype(int)
    if np.sum(n_eval_points) != n_eval:
        n_eval_points[-1] += n_eval - np.sum(n_eval_points)
    piecewise_points = np.zeros((n_eval, 3))
    for k in range(len(points) - 1):
        piecewise_points[
            np.sum(n_eval_points[:k]) : np.sum(n_eval_points[: k + 1]), :
        ] = np.linspace(points[k], points[k + 1], n_eval_points[k] + 1)[:-1, :]

    # --------------------------------- Solve NLP -------------------------------- #
    # Load solver
    nlp_warm = trajectory_generator_solver(n, n_eval, l4c_nerf, warmup=True, threshold=optimization_threshold)

    # solve nlp
    params_flat = piecewise_points.T.flatten()  # update nlp to take this as input!
    sol = nlp_warm["solver"](p=params_flat, lbg=nlp_warm["lbg"], ubg=nlp_warm["ubg"])

    # --------------------------------- Evaluate --------------------------------- #
    # Extract and evaluate solution
    coeffs_warm = np.squeeze(sol["x"]).reshape(2, n).T
    coeffs_warm = np.hstack([np.zeros((n, 1)), coeffs_warm])
    _, f_eval = polynomial(n, n_eval)

    # ---------------------------------------------------------------------------- #
    #                              Collision free NLP                              #
    # ---------------------------------------------------------------------------- #
    # Load solver
    nlp = trajectory_generator_solver(n, n_eval, l4c_nerf, warmup=False, threshold=optimization_threshold)

    # Solve nlp
    x_init = coeffs_warm[:, 1:].T.flatten()
    sol = nlp["solver"](x0=x_init, p=params_flat, lbg=nlp["lbg"], ubg=nlp["ubg"])

    # --------------------------------- Evaluate --------------------------------- #
    # Extract and evaluate solution
    coeffs_sol = np.squeeze(sol["x"]).reshape(2, n).T
    coeffs_sol = np.hstack([np.zeros((n, 1)), coeffs_sol])

    _, f_eval = polynomial(n, n_eval)
    p_eval = np.squeeze(f_eval(coeffs=coeffs_sol)["p"])

    p_sol = p_eval.copy()

    # ---------------------------------------------------------------------------- #
    #                                   Visualize                                  #
    # ---------------------------------------------------------------------------- #

    meshgrid = torch.meshgrid(
        torch.linspace(0, 0, 1),
        torch.linspace(-1.0, 1.2, 200),
        torch.linspace(-0.5, 1, 200),
        indexing='ij'
    )

    points = torch.stack(meshgrid, dim=-1).reshape(-1, 3)
    with torch.no_grad():
        density = model(points).detach()[..., 0]
    points = points.numpy()

    with torch.no_grad():
        density_sol = model(torch.tensor(p_sol, dtype=torch.float32)).detach()[..., 0]

    print(f"Maximum Density in Solution: {density_sol.max()} < Threshold {optimization_threshold:.2f}")

    ax = plt.figure().add_subplot(111)
    ax.plot(p_sol[:, 1], p_sol[:, 2], "-", color=(0.8, 0.12, 0.12), linewidth=3)
    g = ax.scatter(
        points[density > viz_threshold][:, 1],
        points[density > viz_threshold][:, 2],
        cmap="jet",
        c=density[density > viz_threshold],
        s=0.5,
    )
    cb = plt.colorbar(g, ax=ax)
    ax.scatter(p_start[1], p_start[2], color=(0.12, 0.12, 0.8), s=50., zorder=10)
    ax.scatter(p_goal[1], p_goal[2], color=(0.12, 0.8, 0.12), s=50., zorder=10)
    cb.set_label('NeRF Density')
    plt.xticks([], [])
    plt.yticks([], [])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
