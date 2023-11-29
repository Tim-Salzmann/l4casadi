import csv

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from utils import (
    import_velocity_field,
    plot_velocity_field_particle,
    plot_vorticity_particle,
    maybe_mkdir
)


def trajectory_generator_solver(fU, fV, dt, N, u_lim, T, GT):
    # Decision variables and parameters
    x = cs.MX.sym("x", N, 2)
    u = cs.MX.sym("u", N - 1, 2)
    params = cs.MX.sym("params", N + 1, 2)
    x_init = params[0, :]
    x_ref = params[1:, :]

    # Define NLP
    f = 0
    g = []
    lbg = []
    ubg = []
    for k in range(N):
        # cost function
        if k < N - 2:
            f += cs.sum2(u[k + 1, :] - u[k, :]) ** 2

        # spatial bounds
        g = cs.horzcat(g, x[k, :])
        lbg = cs.horzcat(lbg, cs.DM([-0.9, -1.9]).T)
        ubg = cs.horzcat(ubg, cs.DM([7.9, 1.9]).T)

        # stone bounds
        g = cs.horzcat(g, x[k, 0] ** 2 + x[k, 1] ** 2)
        lbg = cs.horzcat(lbg, cs.DM([0.25]).T)
        ubg = cs.horzcat(ubg, cs.DM([1000.0]).T)

        # initial and final states
        if k == 0:
            g = cs.horzcat(g, x[k, :] - x_init)
            lbg = cs.horzcat(lbg, cs.DM([0, 0]).T)
            ubg = cs.horzcat(ubg, cs.DM([0, 0]).T)
        elif k == N - 1:
            g = cs.horzcat(g, x[k, :] - x_ref[k, :])
            lbg = cs.horzcat(lbg, cs.DM([0, 0]).T)
            ubg = cs.horzcat(ubg, cs.DM([0, 0]).T)

        # other states
        if k < N - 1:
            # control limits
            g = cs.horzcat(g, u[k, :])
            lbg = cs.horzcat(lbg, cs.DM([-u_lim, -u_lim]).T)
            ubg = cs.horzcat(ubg, cs.DM([u_lim, u_lim]).T)

            # dynamics continuity
            if GT:
                v_flow = cs.vertcat(fU[k](x[k, :]), fV[k](x[k, :])).T
            else:
                time_stamp = k * T / (N - 1)
                velU = fU(cs.horzcat(time_stamp, x[k, :]))
                velV = fV(cs.horzcat(time_stamp, x[k, :]))
                v_flow = cs.vertcat(velU, velV).T
            # v_flow = cs.DM([0, 0]).T
            x_next = x[k, :] + (v_flow + u[k, :]) * dt  # flow makes the solver fail
            g = cs.horzcat(g, x_next - x[k + 1, :])
            lbg = cs.horzcat(lbg, cs.DM([0, 0]).T)
            ubg = cs.horzcat(ubg, cs.DM([0, 0]).T)

    # Generate solver
    x_nlp = cs.vertcat(cs.reshape(x, N * 2, 1), cs.reshape(u, (N - 1) * 2, 1))
    p_nlp = cs.reshape(params, (N + 1) * 2, 1)
    nlp_dict = {
        "x": x_nlp,
        "f": f,
        "g": g,
        "p": p_nlp,
    }
    nlp_opts = {
        "ipopt.linear_solver": "mumps",
        "ipopt.sb": "yes",
        "ipopt.max_iter": 1000,
        "ipopt.tol": 1e-4,
        "ipopt.print_level": 5,
        "print_time": False,
    }
    nlp_solver = cs.nlpsol("trajectory_generator", "ipopt", nlp_dict, nlp_opts)

    solver = {"solver": nlp_solver, "lbg": lbg, "ubg": ubg}

    return solver


def generate_trajectory():
    # User inputs
    GT = False
    p_start = np.array([7.75, 1.5])
    p_goal = np.array([-0.85, -0.4])
    u_lim = 1
    T = 20

    # -------------------- Import velocity field interpolators ------------------- #
    print("Importing velocity field model...")
    fU, fV = import_velocity_field(GT=GT, device="cpu")
    N = 151  # len(fU)
    dt = T / N
    print("Done.")

    # ------------------------------ Generate solver ----------------------------- #
    print("Generating trajectory generation solver...")
    print("\tT = {}\n\tN = {}\n\tdt = {}".format(T, N, dt))
    nlp = trajectory_generator_solver(fU=fU, fV=fV, dt=dt, N=N, T=T, u_lim=u_lim, GT=GT)
    print("Done.")

    # --------------------------------- Solve NLP -------------------------------- #
    # set initial guess and parameters
    params = np.vstack([p_start, np.tile(p_goal[:, None], N).T])
    u_init = np.zeros((N - 1, 2))
    p_init = np.zeros((N, 2))
    p_init[:, :] = p_start
    x_init = np.vstack([p_init, u_init])

    # solve nlp
    x_init_flat = cs.reshape(x_init, 4 * N - 2, 1)
    params_flat = cs.reshape(params, (N + 1) * 2, 1)
    sol = nlp["solver"](x0=x_init_flat, p=params_flat, lbg=nlp["lbg"], ubg=nlp["ubg"])

    # extract solution
    p_sol = np.squeeze(sol["x"])[: N * 2].reshape(2, N).T
    u_sol = np.squeeze(sol["x"])[N * 2 :].reshape(2, N - 1).T

    # --------------------------------- Visualize -------------------------------- #

    # generate velocity fields
    print("\nGenerating velocity fields for visualization...")
    neval = 25
    Xgrid, Ygrid = np.meshgrid(np.linspace(-1, 8, neval), np.linspace(-2, 2, neval))
    U = np.zeros((N, neval, neval))
    V = np.zeros((N, neval, neval))
    for t in range(0, N):
        for i in range(neval):
            for j in range(neval):
                if GT:
                    U[t, i, j] = np.squeeze(fU[t]([Xgrid[i, j], Ygrid[i, j]]))
                    V[t, i, j] = np.squeeze(fV[t]([Xgrid[i, j], Ygrid[i, j]]))
                else:
                    U[t, i, j] = np.squeeze(fU([t * T / (N - 1), Xgrid[i, j], Ygrid[i, j]]))
                    V[t, i, j] = np.squeeze(fV([t * T / (N - 1), Xgrid[i, j], Ygrid[i, j]]))
    print("Done.")

    print("\nImporting vorticity for visualization ...")
    with open("./data/VORTALL.csv", "r") as file:
        reader = csv.reader(file)
        vorticity = [row for row in reader]
    vorticity = np.array(vorticity, dtype=float)
    print("Done.")

    plt.figure(figsize=(10, 5))
    plot_velocity_field_particle(
        Xgrid, Ygrid, U[0], V[0], p_init[:, 0], p_init[:, 1], p_start, p_goal
    )
    plt.suptitle("Initial guess")

    plt.figure(figsize=(10, 5))
    plot_velocity_field_particle(
        Xgrid, Ygrid, U[0], V[0], p_sol[:, 0], p_sol[:, 1], p_start, p_goal
    )
    plt.suptitle("Minimum energy trajectory")

    plt.figure()
    plt.plot(u_sol)
    plt.plot(-u_lim * np.ones((N - 1, 2)), "r--")
    plt.plot(u_lim * np.ones((N - 1, 2)), "r--")
    plt.ylabel("v [m/s]")
    plt.xlabel("steps")
    plt.suptitle("Control inputs")
    plt.show()

    # ----------------------------- Create animation ----------------------------- #

    path = "./media/"
    maybe_mkdir(path)
    print("\nGenerating animations...")

    # velocity field
    fig, ax = plt.subplots(figsize=(10, 5))
    frames = N
    animation = FuncAnimation(
        fig,
        lambda frame_num: plot_velocity_field_particle(
            Xgrid,
            Ygrid,
            U[frame_num],
            V[frame_num],
            p_sol[max(0, frame_num - 10): frame_num + 1, 0],
            p_sol[max(0, frame_num - 10): frame_num + 1, 1],
            p_start,
            p_goal,
            round(frame_num / frames * T, 3),
        ),
        frames=frames,
        interval=100,
    )
    animation.save(
        path + "trajectory_generation_velocity_field.gif",
        writer="ffmpeg",
    )

    # vorticity field
    fig, ax = plt.subplots(figsize=(10, 5))
    frames = N
    animation = FuncAnimation(
        fig,
        lambda frame_num: plot_vorticity_particle(
            VORT=vorticity[:, frame_num].reshape(449, 199).T,
            x_pcl=p_sol[max(0, frame_num - 10): frame_num + 1, 0],
            y_pcl=p_sol[max(0, frame_num - 10): frame_num + 1, 1],
            p_start=p_start,
            p_goal=p_goal,
            frame_num=round(frame_num / frames * T, 3),
        ),
        frames=frames,
        interval=100,
    )
    animation.save(
        path + "trajectory_generation_vorticity.gif",
        writer="ffmpeg",
    )

    print("Saved animations in:", path)
    plt.show()


if __name__ == '__main__':
    generate_trajectory()

