import csv

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

from utils import plot_velocity_field_particle, maybe_mkdir


def main():
    interpolator_path = "./data/interpolators/"
    maybe_mkdir(interpolator_path)
    maybe_mkdir("./media")
    p_start = np.array([1, 1])
    T = 20

    # -------------------------------- Import data ------------------------------- #
    # import data from csv files
    print("Reading data...")
    with open("./data/UALL.csv", "r") as file:
        reader = csv.reader(file)
        dataU = [row for row in reader]
    dataU = np.array(dataU, dtype=float)

    with open("./data/VALL.csv", "r") as file:
        reader = csv.reader(file)
        dataV = [row for row in reader]
    dataV = np.array(dataV, dtype=float)

    print("Done.")

    # reshape data
    N = dataU.shape[1]  # number of time steps
    dt = T / N
    nx = 449  # grid size in x
    ny = 199  # grid size in y
    x_min = -1
    x_max = 8
    y_min = -2
    y_max = 2

    # ------------------------------- Reshape data ------------------------------- #
    # interpolate to regular grid
    x = np.linspace(x_min, x_max, nx + 1)[:-1]
    y = np.linspace(y_min, y_max, ny + 1)[:-1]

    fU = []
    fV = []
    for t in tqdm(range(0, N), desc="Generating interpolators"):
        U_sample = []
        V_sample = []
        for i in range(0, dataU.shape[1]):
            U_sample.append(dataU[:, i].reshape(nx, ny))
            V_sample.append(dataV[:, i].reshape(nx, ny))
        U_sample = np.squeeze(U_sample)  # [nt, nx, ny]
        V_sample = np.squeeze(V_sample)  # [nt, nx, ny]

        Uvalues = U_sample[t, :, :].ravel(order="F")
        Vvalues = V_sample[t, :, :].ravel(order="F")

        fUt = cs.interpolant("turbulent_flow", "bspline", [x, y], Uvalues)
        fVt = cs.interpolant("turbulent_flow", "bspline", [x, y], Vvalues)

        fUt.save(interpolator_path + "fU_" + str(t) + ".casadi")
        fVt.save(interpolator_path + "fV_" + str(t) + ".casadi")

        fU += [fUt]
        fV += [fVt]

    # ---------------------- Unactuated particle simulation ---------------------- #
    neval = 25
    Xgrid, Ygrid = np.meshgrid(np.linspace(x_min, x_max, neval), np.linspace(y_min, y_max, neval))
    p_particle = np.zeros((N, 2))
    U = np.zeros((N, neval, neval))
    V = np.zeros((N, neval, neval))
    p_particle[0, :] = p_start
    for t in tqdm(range(0, N), desc="Simulating unactuated particle"):
        # update particle
        if t < N - 1:
            v_flow = np.vstack(
                [
                    fU[t]([p_particle[t, 0], p_particle[t, 1]]),
                    fV[t]([p_particle[t, 0], p_particle[t, 1]]),
                ]
            )
            p_particle[t + 1, :] = p_particle[t, :] + dt * v_flow.T

        # get velocity field for all grid
        for i in range(neval):
            for j in range(neval):
                U[t, i, j] = np.squeeze(fU[t]([Xgrid[i, j], Ygrid[i, j]]))
                V[t, i, j] = np.squeeze(fV[t]([Xgrid[i, j], Ygrid[i, j]]))

    # ----------------------------- Create animation ----------------------------- #

    path = "./media/unactuated_velocity_field.gif"
    print("\nGenerating animation...")
    # Create an animation for each frame
    fig, ax = plt.subplots(figsize=(10, 5))
    frames = 90  # N #for N point mass gets out of plot
    animation = FuncAnimation(
        fig,
        lambda frame_num: plot_velocity_field_particle(
            X=Xgrid,
            Y=Ygrid,
            U=U[frame_num],
            V=V[frame_num],
            x_pcl=p_particle[max(0, frame_num - 10): frame_num + 1, 0],
            y_pcl=p_particle[max(0, frame_num - 10): frame_num + 1, 1],
            frame_num=round(frame_num / frames * T, 3),
        ),
        frames=frames,
        interval=100,
    )
    animation.save(
        path,
        writer="ffmpeg",
    )

    print("Saved animation in:", path)
    plt.show()


if __name__ == '__main__':
    main()
