import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from utils import (
    import_interpolators, maybe_mkdir
)


def normalize_dataset(x, y, mean, std):
    mean_vals_x = mean['x']
    std_vals_x = std['x']
    x = (x - mean_vals_x) / std_vals_x

    mean_vals_y = mean['y']
    std_vals_y = std['y']
    y = (y - mean_vals_y) / std_vals_y

    return x, y


def get_dataset_stats(x, y):
    mean_vals_x = np.mean(x, axis=0)
    std_vals_x = np.std(x, axis=0)

    mean_vals_y = np.mean(y, axis=0)
    std_vals_y = np.std(y, axis=0)

    mean = {"x": mean_vals_x, "y": mean_vals_y}
    std = {"x": std_vals_x, "y": std_vals_y}
    return mean, std


def generate_dataset(fU, fV, T, ngrid):
    N = len(fU)

    Xgrid, Ygrid = np.meshgrid(np.linspace(-0.9, 7.9, ngrid), np.linspace(-1.9, 1.9, ngrid))

    X = np.zeros((N * (ngrid * ngrid), 3))
    Y = np.zeros((N * (ngrid * ngrid), 2))
    cont = 0
    for t in tqdm(range(0, N), desc="Generating dataset", disable=False):
        for i in range(ngrid):
            for j in range(ngrid):
                u = np.squeeze(fU[t]([Xgrid[i, j], Ygrid[i, j]]))
                v = np.squeeze(fV[t]([Xgrid[i, j], Ygrid[i, j]]))
                X[cont, :] = np.array([t * T / (N - 1), Xgrid[i, j], Ygrid[i, j]])  #
                Y[cont, :] = np.array([u, v])
                cont += 1

    return X, Y


def generate_val_dataset(fU, fV, T, ngrid):
    N = len(fU)
    Xgrid, Ygrid = np.meshgrid(np.random.uniform(-0.9, 7.9, size=(ngrid,)), np.random.uniform(-1.9, 1.9, size=(ngrid,)))

    X = np.zeros((N * (ngrid * ngrid), 3))
    Y = np.zeros((N * (ngrid * ngrid), 2))
    cont = 0
    for t in tqdm(range(0, N), desc="Generating dataset", disable=False):
        for i in range(ngrid):
            for j in range(ngrid):
                u = np.squeeze(fU[t]([Xgrid[i, j], Ygrid[i, j]]))
                v = np.squeeze(fV[t]([Xgrid[i, j], Ygrid[i, j]]))
                X[cont, :] = np.array([t * T / (N - 1), Xgrid[i, j], Ygrid[i, j]])  #
                Y[cont, :] = np.array([u, v])
                cont += 1

    return X, Y


def train():
    file_name = "turbolent_flow_model.pt"
    maybe_mkdir("./models/")
    # Problem specific parameters
    T = 20
    ngrid_train = 200
    ngrid_valid = ngrid_train

    # Learning parameters
    n_epochs = 10
    layer_size = 256
    learning_rate = 1e-3
    batch_size = 1024

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------- Import ground truth interpolators -------------------- #

    print("Importing velocity field interpolators...")
    fU, fV = import_interpolators()
    N = len(fU)
    print("Done.\n")

    # ----------------------------- Generate datasets ---------------------------- #

    print("Generating training dataset...")
    x, y = generate_dataset(
        fU=fU, fV=fV, T=T, ngrid=ngrid_train
    )
    mean, std = get_dataset_stats(x, y)
    x, y = normalize_dataset(x, y, mean, std)
    dataset_train = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).float())
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    print("Generating validation dataset...")
    x, y = generate_val_dataset(
        fU=fU, fV=fV, T=T, ngrid=ngrid_valid
    )
    x, y = normalize_dataset(x, y, mean, std)
    dataset_valid = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).float())
    dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True)

    # ----------------------------------- Model ---------------------------------- #

    model = torch.nn.Sequential(
        torch.nn.Linear(3, layer_size),
        torch.nn.GELU(),
        torch.nn.Linear(layer_size, layer_size),
        torch.nn.GELU(),
        torch.nn.Linear(layer_size, layer_size),
        torch.nn.GELU(),
        torch.nn.Linear(layer_size, 2),
    )

    # ------------------------ Optimizer and loss function ----------------------- #
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    loss_func = torch.nn.L1Loss()
    val_loss_func = torch.nn.L1Loss()

    n_train = ngrid_train * ngrid_train * N
    n_train_iter = int(np.ceil(n_train / batch_size))

    model = model.to(device)
    n_steps = 0
    for epoch in range(n_epochs):
        # --------------------------------- Training --------------------------------- #
        model.train()
        progress = tqdm(dataloader_train)
        for i, (X_tr, Y_tr) in enumerate(progress):
            X_tr = X_tr.to(device)
            Y_tr = Y_tr.to(device)
            # train step
            optimizer.zero_grad()  # clear the gradients
            y_pred = model(X_tr)  # forward pass
            loss = loss_func(y_pred, Y_tr)  # calculate loss
            loss.backward()  # backward pass
            optimizer.step()
            train_loss = loss.item()  # extract loss

            # log statistics
            progress.set_description(f"Epoch {epoch+1}/{n_epochs}, Train Loss {train_loss:.4f}")
            n_steps += 1

        if epoch % 2 != 0:
            continue
        # -------------------------------- Validation -------------------------------- #
        model.eval()
        val_losses = []
        progress = tqdm(dataloader_valid)
        with torch.no_grad():
            for i, (x_valid, y_valid) in enumerate(progress):
                x_valid = x_valid.to(device)
                y_valid = y_valid.to(device)
                # valid step
                y_pred = model(x_valid)  # forward pass
                loss = val_loss_func(y_pred, y_valid)  # calculate loss
                valid_loss = loss.item()  # extract loss

                val_losses.append(valid_loss)

                # print statistics
                progress.set_description(f"Epoch {epoch+1}/{n_epochs}, Val Loss {valid_loss:.6f}")

            # save model
            checkpoint = {
                "model": model,
                "mean": mean,
                "std": std,
            }
            torch.save(
                checkpoint, "./models/" + file_name
            )


if __name__ == '__main__':
    torch.manual_seed(123)
    train()

