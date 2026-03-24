import csv
import os
import sys
from multiprocessing import freeze_support

import matplotlib.pyplot as plt
from pysr import PySRRegressor
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from equation_search import equation_search


<<<<<<< HEAD
def load_csv(path: str) -> tuple[torch.Tensor, torch.Tensor]:
=======
def load_csv(path):
>>>>>>> add test cases on deepchem data
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        x_rows = []
        y_vals = []
        for row in reader:
<<<<<<< HEAD
            x_rows.append([float(row['v'])])
            y_vals.append(float(row['L']))
=======
            x_rows.append([float(row["v"])])
            y_vals.append(float(row["L"]))
>>>>>>> add test cases on deepchem data

    X = torch.tensor(x_rows, dtype=torch.float32)
    y = torch.tensor(y_vals, dtype=torch.float32)
    return X, y


<<<<<<< HEAD
def mse_torch(yhat: torch.Tensor, y: torch.Tensor) -> float:
    return float(((yhat - y) ** 2).mean().item())


def write_results(name: str, es_eq: str, es_loss: float, pysr_eq: str, pysr_loss: float) -> None:
=======
def mse_torch(yhat, y):
    return float(((yhat - y) ** 2).mean().item())


def write_results(name, es_eq, es_loss, pysr_eq, pysr_loss):
>>>>>>> add test cases on deepchem data
    out_dir = os.path.join(REPO_ROOT, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "results.csv")
    header_needed = not os.path.exists(out_path)
    with open(out_path, "a", newline="") as f:
        w = csv.writer(f)
        if header_needed:
<<<<<<< HEAD
            w.writerow(["dataset", "equation_search_equation", "equation_search_loss", "pysr_equation", "pysr_loss"])
        w.writerow([name, es_eq, es_loss, pysr_eq, pysr_loss])


def main() -> None:
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
        w.writerow([name, es_eq, es_loss, pysr_eq, pysr_loss])


def main():
>>>>>>> add test cases on deepchem data
    csv_path = os.path.join(os.path.dirname(__file__), "tully.csv")
    dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
    X, y = load_csv(csv_path)

    # Custom equation search
    state = equation_search(X, y, niterations=300)
    best_overall = min(state.halls_of_fame[0].values(), key=lambda c: c.cost)
    es_yhat = best_overall.tree.forward(X)
    es_loss = mse_torch(es_yhat, y)
    es_eq = best_overall.tree.to_string()
    print("EquationSearch equation:", es_eq)
    print("EquationSearch loss:", es_loss)

    # PySR fit
    X_np = X.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    model = PySRRegressor(
        niterations=300,
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

    # Plot data + fits
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
    plt.title("Tully-Fisher Data: equation_search vs PySR")
    plt.xlabel("v")
    plt.ylabel("L")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.join(REPO_ROOT, "outputs"), exist_ok=True)
    plt.savefig(os.path.join(REPO_ROOT, "outputs", "tully_compare.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    freeze_support()
    main()
