import csv
import os

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
import torch

import l4casadi as l4c


def maybe_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# ------------------------------- Visualization ------------------------------ #
def plot_vorticity(VORT, frame_num=None):
    # import colormap
    with open("navigation_turbulent_flow/data/CC.csv", "r") as file:
        reader = csv.reader(file)
        CC = [row for row in reader]
    CC = np.array(CC, dtype=float)
    custom_cmap = plt.matplotlib.colors.ListedColormap(CC)

    # saturate vorticity
    vortmin = -5
    vortmax = 5
    VORT[VORT > vortmax] = vortmax
    VORT[VORT < vortmin] = vortmin

    # plot vorticity
    plt.clf()
    plt.gca().invert_xaxis()
    plt.imshow(
        np.flipud(VORT),  # VORT
        extent=[0, VORT.shape[1], 0, VORT.shape[0]],
        cmap=custom_cmap,
        vmin=vortmin,
        vmax=vortmax,
    )

    plt.xticks(
        [1, 50, 100, 150, 200, 250, 300, 350, 400, 449],
        ["-1", "0", "1", "2", "3", "4", "5", "6", "7", "8"],
    )
    plt.yticks([1, 50, 100, 150, 199], ["2", "1", "0", "-1", "-2"])
    plt.gca().set_aspect("equal", adjustable="box")

    # add contour lines (positive = solid, negative = dotted)
    levels_neg = np.array(
        [-5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, -0.25, -0.125]
    )
    levels_pos = np.array([0.125, 0.25, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5])
    plt.contour(
        VORT, levels=levels_neg, colors="k", linestyles=":", linewidths=1.2, zorder=1
    )
    plt.contour(VORT, levels=levels_pos, colors="k", linewidths=1.2, zorder=1)

    # place cylinder
    theta = np.linspace(0, 2 * np.pi, 100)
    x = 49 + 25 * np.sin(theta)
    y = 99 + 25 * np.cos(theta)
    plt.fill(x, y, color="gray", alpha=1.0, zorder=2)  # place cylinder
    plt.plot(x, y, "k", linewidth=1.2, zorder=2)  # cylinder boundary

    plt.xticks([], [])
    plt.yticks([], [])
    plt.tight_layout()

    #plt.gca().set_position([0, 0, 1, 1])  # remove whitespace around the image
    if frame_num is not None:
        plt.title(f"Frame {frame_num}")

    # plt.savefig("cylinder.png", bbox_inches="tight")
    # plt.show()


def plot_vorticity_particle(VORT, x_pcl, y_pcl, p_start, p_goal, frame_num=None):
    # import colormap
    with open("./data/CC.csv", "r") as file:
        reader = csv.reader(file)
        CC = [row for row in reader]
    CC = np.array(CC, dtype=float)
    custom_cmap = plt.matplotlib.colors.ListedColormap(CC)

    # saturate vorticity
    vortmin = -5
    vortmax = 5
    VORT[VORT > vortmax] = vortmax
    VORT[VORT < vortmin] = vortmin

    # plot vorticity
    plt.clf()
    plt.imshow(
        np.flipud(VORT),  # VORT
        extent=[0, VORT.shape[1], 0, VORT.shape[0]],
        cmap=custom_cmap,
        vmin=vortmin,
        vmax=vortmax,
    )

    plt.xticks(
        [1, 50, 100, 150, 200, 250, 300, 350, 400, 449],
        ["-1", "0", "1", "2", "3", "4", "5", "6", "7", "8"],
    )
    plt.yticks([1, 50, 100, 150, 199], ["2", "1", "0", "-1", "-2"])
    plt.gca().set_aspect("equal", adjustable="box")

    # add contour lines (positive = solid, negative = dotted)
    levels_neg = np.array(
        [-5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, -0.25, -0.125]
    )
    levels_pos = np.array([0.125, 0.25, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5])
    plt.contour(
        VORT, levels=levels_neg, colors="k", linestyles=":", linewidths=1.2, zorder=1
    )
    plt.contour(VORT, levels=levels_pos, colors="k", linewidths=1.2, zorder=1)

    # place cylinder
    theta = np.linspace(0, 2 * np.pi, 100)
    x = 49 + 25 * np.sin(theta)
    y = 99 + 25 * np.cos(theta)
    plt.fill(x, y, color="gray", alpha=1.0, zorder=2)  # place cylinder
    plt.plot(x, y, "k", linewidth=1.2, zorder=2)  # cylinder boundary

    # plot start and goal
    plt.plot((p_start[0] + 1) * 449 / 9, (p_start[1] + 2) * 199 / 4, "go")
    plt.plot((p_goal[0] + 1) * 449 / 9, (p_goal[1] + 2) * 199 / 4, "ro")

    # plot particle
    plt.plot(
        (x_pcl[-1] + 1) * (449 / 9),
        (y_pcl[-1] + 2) * (199 / 4),
        "o",
        color="magenta",
        markersize=12,
        markeredgecolor="black",
        zorder=3,
    )
    plt.plot(
        (x_pcl + 1) * (449 / 9),
        (y_pcl + 2) * (199 / 4),
        "-",
        color="magenta",
        zorder=2,
    )
    plt.gca().invert_xaxis()
    plt.xticks([], [])
    plt.yticks([], [])
    plt.tight_layout()

    # set title
    if frame_num is not None:
        #plt.gca().set_position([0, 0, 1, 1])
        plt.title(f"Time {frame_num:.1f} s")


def plot_velocity_field(X, Y, U, V, ax, cbar=False):
    quiver = ax.quiver(X, Y, -U, V, np.sqrt(U**2 + V**2), cmap="turbo")   # -U to flip x-axis
    if cbar:
        cbar = plt.colorbar(quiver, ax=ax)
        cbar.set_label("Velocity [m/s]")

    ax.set_aspect("equal")

    theta = np.linspace(0, 2 * np.pi, 100)
    x = (25 * np.sin(theta)) * 4 / 199
    y = (25 * np.cos(theta)) * 9 / 449
    ax.fill(x, y, color="gray")  # place cylinder
    ax.plot(x, y, "k", linewidth=1.2)  # cylinder boundary
    ax.invert_xaxis()
    ax.xticks([], [])
    ax.yticks([], [])
    ax.tight_layout()


def plot_velocity_field_particle(
    X, Y, U, V, x_pcl, y_pcl, p_start=None, p_goal=None, frame_num=None
):
    plt.clf()
    plt.gca().invert_xaxis()

    plt.quiver(X, Y, -U, V, np.sqrt(U**2 + V**2), cmap="turbo")  # -U to flip x-axis
    plt.gca().set_aspect("equal")

    plt.plot(
        x_pcl[-1],
        y_pcl[-1],
        "o-",
        color="magenta",
        markersize=12,
        markeredgecolor="black",
        zorder=3,
    )
    plt.plot(x_pcl, y_pcl, "-", color="magenta", zorder=2)

    theta = np.linspace(0, 2 * np.pi, 100)
    x = (25 * np.sin(theta)) * 4 / 199
    y = (25 * np.cos(theta)) * 9 / 449
    plt.fill(x, y, color="gray")  # place cylinder
    plt.plot(x, y, "k", linewidth=1.2)  # cylinder boundary

    plt.xticks([], [])
    plt.yticks([], [])
    plt.tight_layout()

    if p_goal is not None:
        plt.plot(p_start[0], p_start[1], "go")
        plt.plot(p_goal[0], p_goal[1], "ro")

    if frame_num is not None:
        plt.title(f"Time {frame_num:.1f} s")


def compare_velocity_fields(t, Xgrid, Ygrid, uGT, vGT, uNN, vNN):
    plt.clf()
    fig = plt.gcf()
    fig.set_size_inches(10, 10)

    ax = fig.add_subplot(311)
    plot_velocity_field(X=Xgrid, Y=Ygrid, U=uGT, V=vGT, ax=ax, cbar=True)
    ax.set_title("Ground truth")
    ax.invert_xaxis()

    ax = fig.add_subplot(312)
    plot_velocity_field(X=Xgrid, Y=Ygrid, U=uNN, V=vNN, ax=ax, cbar=True)
    ax.set_title("Neural Network model")
    ax.invert_xaxis()

    ax = fig.add_subplot(313)
    plot_velocity_field(X=Xgrid, Y=Ygrid, U=uNN - uGT, V=vNN - vGT, ax=ax, cbar=True)
    ax.set_title("Residual")
    ax.invert_xaxis()

    plt.suptitle(f"Time {t:.1f} s")


# ---------------------------------- Others ---------------------------------- #
def import_interpolators():
    interp_path = "./data/interpolators/"
    n_interpolators = 151

    fU_interp = []
    fV_interp = []
    for i in range(n_interpolators):
        fU_interp += [cs.Function.load(interp_path + "fU_" + str(i) + ".casadi")]
        fV_interp += [cs.Function.load(interp_path + "fV_" + str(i) + ".casadi")]

    return fU_interp, fV_interp


def import_l4casadi_model(device):
    checkpoint = torch.load(
        "./models/turbolent_flow_model.pt",
        map_location=torch.device(device),
    )

    model = checkpoint["model"]  # .to(device)
    meanX = checkpoint["mean"]["x"]
    stdX = checkpoint["std"]["x"]
    meanY = checkpoint["mean"]["y"]
    stdY = checkpoint["std"]["y"]
    T = 20

    x = cs.MX.sym("x", 3)
    xn = (x - meanX) / stdX

    y = l4c.L4CasADi(model, name="turbulent_model", generate_adj1=False, generate_jac_jac=True)(xn.T).T
    y = y * stdY + meanY
    fU = cs.Function("fU", [x], [y[0]])
    fV = cs.Function("fV", [x], [y[1]])

    return fU, fV, T


def import_velocity_field(GT, device=None):
    if GT:
        fU, fV = import_interpolators()
    else:
        fU, fV, _ = import_l4casadi_model(device=device)

    return fU, fV
