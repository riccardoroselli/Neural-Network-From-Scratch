# training/gridsearch.py
from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .config import expand_grid, load_config
from .trainer import Trainer

# You will provide these from your codebase when calling run_grid:
# - build_model_fn(run_cfg) -> Model
# - load_data_fn(run_cfg) -> (X_train, y_train, X_val, y_val)
# The gridsearch module stays framework-agnostic.


@dataclass
class GridResult:
    run_id: str
    config: Dict[str, Any]
    seed: int
    best_epoch: Optional[int]
    best_val_loss: Optional[float]
    best_val_acc: Optional[float]
    final_val_loss: Optional[float]
    final_val_acc: Optional[float]
    train_time_sec: float


def _hash_config(cfg: Dict[str, Any]) -> str:
    # Stable hash for a config dict (JSON canonical form)
    s = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _extract_history_best(history_dict: Dict[str, List[Any]], monitor: str, mode: str) -> Tuple[Optional[int], Optional[float]]:
    """
    Returns (best_epoch_index_1based, best_value) for a monitored history series.
    """
    series = history_dict.get(monitor, None)
    if not series:
        return None, None

    values = np.array(series, dtype=float)
    if mode == "min":
        idx0 = int(np.argmin(values))
    elif mode == "max":
        idx0 = int(np.argmax(values))
    else:
        raise ValueError("mode must be 'min' or 'max'")
    return idx0 + 1, float(values[idx0])


def _infer_mode_from_monitor(monitor: str) -> str:
    m = monitor.lower()
    if "acc" in m or "accuracy" in m:
        return "max"
    return "min"


def _get_metric_key(history_dict: Dict[str, List[Any]]) -> Optional[str]:
    """
    Your metric class names become keys in logs, e.g. "Accuracy". :contentReference[oaicite:6]{index=6}
    We pick Accuracy if present; otherwise return first non-loss metric.
    """
    if "Accuracy" in history_dict:
        return "Accuracy"
    # any other metric key
    for k in history_dict.keys():
        if k not in ("loss", "val_loss"):
            return k
    return None


def run_grid(
    config_path: str,
    build_model_fn: Callable[[Dict[str, Any]], Any],
    load_data_fn: Callable[[Dict[str, Any]], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    out_csv_path: str = "results/grid_runs.csv",
    seeds: Optional[List[int]] = None,
    objective: str = "val_loss",
    objective_mode: str = "auto",
    verbose: int = 0,
) -> Dict[str, Any]:
    """
    Run a grid search.

    Parameters
    ----------
    config_path : str
        Path to YAML/JSON with {base: ..., grid: ...}
    build_model_fn : (run_cfg) -> Model
        User-supplied: creates a fresh model for each run.
    load_data_fn : (run_cfg) -> (X_train, y_train, X_val, y_val)
        User-supplied: loads and returns train/val split arrays.
    out_csv_path : str
        Where to write results.
    seeds : list[int] | None
        If None, uses [base.seed] or [0].
    objective : str
        Which metric to rank configs by (e.g. "val_loss" or "val_Accuracy").
    objective_mode : str
        "min"/"max"/"auto". Auto infers from objective name.
    verbose : int
        0 silent, 1 prints progress.

    Returns
    -------
    dict with:
      - best_config
      - best_score
      - results (list of rows)
    """
    cfg = load_config(config_path)
    run_cfgs = expand_grid(cfg)

    # seeds
    if seeds is None:
        base_seed = cfg.get("base", {}).get("seed", 0)
        seeds = [int(base_seed)]

    if objective_mode == "auto":
        objective_mode = _infer_mode_from_monitor(objective)
    if objective_mode not in ("min", "max"):
        raise ValueError("objective_mode must be 'min', 'max', or 'auto'.")

    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)

    # CSV header
    header = [
        "run_id",
        "seed",
        "objective",
        "objective_mode",
        "best_epoch",
        "best_val_loss",
        "best_val_acc",
        "final_val_loss",
        "final_val_acc",
        "train_time_sec",
        "config_json",
    ]
    write_header = not os.path.exists(out_csv_path)

    all_rows: List[Dict[str, Any]] = []

    best_overall = None  # (score, row)

    with open(out_csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()

        total = len(run_cfgs) * len(seeds)
        done = 0

        for run_cfg in run_cfgs:
            run_id = _hash_config(run_cfg)

            for seed in seeds:
                done += 1
                if verbose:
                    print(f"[grid] {done}/{total} run_id={run_id} seed={seed}")

                np.random.seed(int(seed))  # reproducibility in your init/dropout paths

                model = build_model_fn(run_cfg)
                trainer = Trainer(model, verbose=0)  # keep grid output clean

                X_train, y_train, X_val, y_val = load_data_fn(run_cfg)

                training_cfg = run_cfg.get("training", {})
                epochs = int(training_cfg.get("epochs", 200))
                batch_size = int(training_cfg.get("batch_size", 32))
                shuffle = bool(training_cfg.get("shuffle", True))

                t0 = time.time()
                history = trainer.fit(
                    X_train, y_train,
                    X_val=X_val, y_val=y_val,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    seed=int(seed),
                    include_reg_in_val=bool(training_cfg.get("include_reg_in_val", False)),
                )
                t1 = time.time()

                h = history.to_dict()

                # Best val_loss and val_Accuracy if present
                best_epoch, best_val_loss = _extract_history_best(h, "val_loss", "min")

                metric_key = _get_metric_key(h)  # e.g., "Accuracy"
                best_val_acc = None
                final_val_acc = None
                if metric_key is not None and f"val_{metric_key}" in h:
                    _, best_val_acc = _extract_history_best(h, f"val_{metric_key}", "max")
                    final_val_acc = _safe_float(h[f"val_{metric_key}"][-1])

                final_val_loss = _safe_float(h["val_loss"][-1]) if "val_loss" in h else None

                # objective score
                if objective in h:
                    obj_series = h[objective]
                    obj_value = float(obj_series[-1])
                else:
                    # common case: objective is "val_loss" or "val_Accuracy"
                    obj_value = float(h.get(objective, [np.inf])[-1])

                row = {
                    "run_id": run_id,
                    "seed": int(seed),
                    "objective": objective,
                    "objective_mode": objective_mode,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "best_val_acc": best_val_acc,
                    "final_val_loss": final_val_loss,
                    "final_val_acc": final_val_acc,
                    "train_time_sec": float(t1 - t0),
                    "config_json": json.dumps(run_cfg, sort_keys=True),
                }

                writer.writerow(row)
                all_rows.append(row)

                # select best overall by objective
                score = _safe_float(obj_value)
                if score is None:
                    continue

                if best_overall is None:
                    best_overall = (score, row)
                else:
                    best_score = best_overall[0]
                    if objective_mode == "min":
                        if score < best_score:
                            best_overall = (score, row)
                    else:
                        if score > best_score:
                            best_overall = (score, row)

    if best_overall is None:
        return {"best_config": None, "best_score": None, "results": all_rows}

    best_score, best_row = best_overall
    best_cfg = json.loads(best_row["config_json"])

    return {"best_config": best_cfg, "best_score": best_score, "results": all_rows}
