import csv
import os
import sys
from multiprocessing import freeze_support
<<<<<<< HEAD
import matplotlib.pyplot as plt
from pysr import PySRRegressor
import torch
from EquationSearch import equation_search
=======

import matplotlib.pyplot as plt
from pysr import PySRRegressor
import torch
>>>>>>> add test cases on deepchem data

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

<<<<<<< HEAD
=======
from equation_search import equation_search
>>>>>>> add test cases on deepchem data


def load_hubble_csv(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        d_vals = []
        v_vals = []
        for row in reader:
            d_vals.append(float(row["D"]))
            v_vals.append(float(row["v"]))

    X = torch.tensor(d_vals, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(v_vals, dtype=torch.float32)
    return X, y


def mse_torch(yhat, y):
    return float(((yhat - y) ** 2).mean().item())


def write_results(name, es_eq, es_loss, pysr_eq, pysr_loss):
    out_dir = os.path.join(REPO_ROOT, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "results.csv")
    header_needed = not os.path.exists(out_path)
    with open(out_path, "a", newline="") as f:
        w = csv.writer(f)
        if header_needed:
<<<<<<< HEAD
            w.writerow(["dataset", "equation_search_equation", "equation_search_loss", "pysr_equation", "pysr_loss"])
=======
            w.writerow(
                [
                    "dataset",
                    "equation_search_equation",
                    "equation_search_loss",
                    "pysr_equation",
                    "pysr_loss",
                ]
            )
>>>>>>> add test cases on deepchem data
        w.writerow([name, es_eq, es_loss, pysr_eq, pysr_loss])


def main():
    csv_path = os.path.join(os.path.dirname(__file__), "hubble.csv")
    dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
    X, y = load_hubble_csv(csv_path)
<<<<<<< HEAD
    state = equation_search(X, y, niterations=250, nout=1)

    best_overall = min(state.hof[0].values(), key=lambda c: c.cost)
=======
    state = equation_search(X, y, niterations=1000)

    best_overall = min(state.halls_of_fame[0].values(), key=lambda c: c.cost)
>>>>>>> add test cases on deepchem data
    es_yhat = best_overall.tree.forward(X)
    es_loss = mse_torch(es_yhat, y)
    es_eq = best_overall.tree.to_string()
    print("EquationSearch equation:", es_eq)
    print("EquationSearch loss:", es_loss)

    X_np = X.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    model = PySRRegressor(
<<<<<<< HEAD
        niterations=250,
=======
        niterations=1000,
>>>>>>> add test cases on deepchem data
        populations=8,
        maxsize=30,
        batching=False,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos", "neg", "exp", "log"],
        loss="loss(x, y) = (x - y)^2",
    )
    model.fit(X_np, y_np)
    pysr_pred = model.predict(X_np)
    pysr_loss = float(((pysr_pred - y_np) ** 2).mean())
    pysr_eq = str(model.get_best())
    print("PySR best equation:")
    print(pysr_eq)
    print("PySR loss:", pysr_loss)
    write_results(dataset_name, es_eq, es_loss, pysr_eq, pysr_loss)

    x_min, x_max = X[:, 0].min().item(), X[:, 0].max().item()
    x_grid = torch.zeros((200, X.shape[1]), dtype=X.dtype)
    x_grid[:, 0] = torch.linspace(x_min, x_max, 200)
    if X.shape[1] > 1:
        x_means = X.mean(dim=0)
        x_grid[:, 1:] = x_means[1:]

    es_line = best_overall.tree.forward(x_grid).detach().cpu().numpy()
    pysr_line = model.predict(x_grid.detach().cpu().numpy())

    plt.figure(figsize=(7, 5))
    plt.scatter(X_np[:, 0], y_np, s=25, alpha=0.8, label="data")
    plt.plot(x_grid[:, 0].numpy(), es_line, label="equation_search")
    plt.plot(x_grid[:, 0].numpy(), pysr_line, label="PySR")
    plt.title("Hubble Data: equation_search vs PySR")
    plt.xlabel("Distance (D)")
    plt.ylabel("Velocity (v)")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.join(REPO_ROOT, "outputs"), exist_ok=True)
    plt.savefig(os.path.join(REPO_ROOT, "outputs", "hubble_compare.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    freeze_support()
    main()
