"""optuna.py – hyper‑parameter search for FlexibleNN on many datasets
================================================================
Runs Optuna separately for each **{dataset × qubit‑count}** pair.
Data‑specific objects (in_dim, loaders, …) are captured in a *factory*
that returns an `objective(trial)` closure, so Optuna receives exactly
one positional argument (`trial`).

On completion, the best hyper‑parameters for every pair are written to
``best_flexiblenn_params_<dataset>_<nq>.json``.
"""

from __future__ import annotations

import time
import json, pathlib, torch, optuna
from typing import Dict, List, Tuple

from utils.data.preprocessing import data_pipeline
from utils.nn.ClassicalNN import FlexibleNN

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
DATASETS = {  # OpenML dataset IDs and their feature counts
    61: 4,  # Iris
    187: 13,  # Wine
    37: 8,  # Diabetes
}
N_QUBITS = [2, 4, 6, 8]  # qubit counts to test
N_LAYERS = [1, 2, 3, 4, 5]  # circuit depths to test

BATCH_SIZE: int = 32
SEED: int = 42
MAX_EPOCHS: int = 100
PATIENCE: int = 10
N_TRIALS: int = 50
TIMEOUT: int = 3600  # seconds


def _train_one_epoch(model: FlexibleNN, loader: torch.utils.data.DataLoader, opt):
    model.train()
    for xb, yb in loader:
        model._train_batch(xb, yb, opt)


@torch.no_grad()
def get_accuracy(model: FlexibleNN, loader: torch.utils.data.DataLoader) -> float:
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        logits, _ = model._evaluate_batch_loss_and_logits(xb, yb)
        correct += (logits.argmax(1) == yb).sum().item()
        total += yb.size(0)
    return correct / total


def make_objective(
    in_dim: int,
    out_dim: int,
    layers: int,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
):
    """Return an Optuna objective closure bound to this dataset.
    We record additional metrics (val_loss) in `trial.user_attrs` so they
    can be dumped later together with hyper‑parameters.
    """

    ce_loss = torch.nn.CrossEntropyLoss()

    def _val_loss(model: FlexibleNN) -> float:
        model.eval()
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                total_loss += ce_loss(logits, yb).item() * yb.size(0)
                n += yb.size(0)
        return total_loss / n

    def objective(trial: optuna.Trial) -> Tuple[float, float]:
        # hyper‑parameter search space
        # width = trial.suggest_int("width", 2, 64, step=2)
        act = trial.suggest_categorical("act", ["relu"])
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        wd = trial.suggest_float("wd", 1e-6, 1e-3, log=True)
        lam_cond = trial.suggest_float("lam_cond", 0.0, 1e-3)

        model = FlexibleNN(
            in_dim,
            hidden_dims=[in_dim] * layers, # "barrel" layers, i.e. same width (e.g. [64, 64, 64])
            output_dim=out_dim,
            act=act,
            condition_number=lam_cond,
            scale_on_export=False,
        )
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        best_val_acc, stale = 0.0, 0
        for _ in range(MAX_EPOCHS):
            _train_one_epoch(model, train_loader, opt)
            val_acc = get_accuracy(model, val_loader)
            train_acc = get_accuracy(model, train_loader)
            if val_acc > best_val_acc + 1e-4:
                best_val_acc, stale = val_acc, 0
            else:
                stale += 1
            if stale >= PATIENCE:
                break

        train_acc = get_accuracy(model, train_loader)
        gen_gap = abs(train_acc - best_val_acc)

        wb = model.get_weights_and_biases(model)
        max_abs = max(t.abs().max().item() for t in wb.values())

        val_loss = _val_loss(model)

        trial.set_user_attr("max_abs", max_abs)
        trial.set_user_attr("gen_gap", gen_gap)
        trial.set_user_attr("val_loss", val_loss)

        return val_acc

    return objective


def _dump_params(meta: Dict[str, object], path: pathlib.Path):
    """Write metrics + hyper‑parameters to JSON."""
    path.write_text(json.dumps(meta, indent=2))
    print(f"  → saved to {path}")


def main(
    dataset_feature_counts: Dict[int, int], n_qubits: List[int], n_layers: List[int]
) -> None:
    
    for ds_id, n_feat in dataset_feature_counts.items():
        for layers in n_layers:
            for nq in n_qubits:
                if nq > n_feat:
                    print(f"[skip] dataset {ds_id}: {n_feat} features < {nq} qubits")
                    continue
                classical, _, in_dim, out_dim, _ = data_pipeline(
                    openml_dataset_id=ds_id,
                    batch_size=BATCH_SIZE,
                    seed=SEED,
                    do_pca=True,
                    n_components=nq,
                )
                train_loader, val_loader, _ = classical[6:9]

                study = optuna.create_study(
                    # directions=["maximize","minimize"],
                    directions=["maximize"],
                    sampler=optuna.samplers.TPESampler(seed=SEED),
                    pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
                )
                study.optimize(
                    make_objective(in_dim, out_dim, layers, train_loader, val_loader),
                    n_trials=N_TRIALS,
                    timeout=TIMEOUT,
                )

                best = study.best_trials[0]
                print(
                    f"\nDataset {ds_id} · {nq} qubits  —  val‑acc={best.values[0]:.3f}"
                )  # max|w|={best.values[1]:.3g}")
                for k, v in best.params.items():
                    print(f"    {k} = {v}")

                # save best hyper‑parameters
                results_dir = pathlib.Path("results/optuna")
                results_dir.mkdir(parents=True, exist_ok=True)
                out_json = (
                    results_dir / f"best_flexiblenn_params_{ds_id}_{nq}_{layers}.json"
                )
                meta = dict(best.params)  # hyper‑params
                meta.update(
                    {
                        "val_acc": best.values[0],
                        "gen_gap": best.user_attrs.get("gen_gap"),
                        "val_loss": best.user_attrs.get("val_loss"),
                        "max_abs": best.user_attrs.get("max_abs"),
                        "dataset_id": ds_id,
                        "n_qubits": nq,
                        "layers": layers,
                    }
                )
                _dump_params(meta, out_json)

                # # export angles
                # model = FlexibleNN(
                #     in_dim,
                #     hidden_dims=[best.params["width"]]*best.params["n_layers"],
                #     output_dim=out_dim,
                #     act=best.params["act"],
                #     condition_number=best.params["lam_cond"],
                #     scale_on_export=True,
                # )
                # angles = model.export_weights()


if __name__ == "__main__":
    start_time = time.time()    
    main(DATASETS, N_QUBITS, N_LAYERS)
    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time:.4f} seconds")
    print("Done.")