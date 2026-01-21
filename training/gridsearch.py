
# training/gridsearch.py
import csv
import hashlib
import json
import os
import time

import numpy as np

from .config import expand_grid, load_config
from .trainer import Trainer
from .holdout_cv import holdout_validation
from .kfold_cv import kfold_cross_validation


def _hash_config(cfg):
    s = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def _safe_float(x):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _infer_mode_from_monitor(monitor):
    m = monitor.lower()
    if "acc" in m or "accuracy" in m:
        return "max"
    return "min"


def _get_val_acc_key(metrics_dict):
    if "Accuracy" in metrics_dict:
        return "Accuracy"
    for k in metrics_dict.keys():
        if k != "loss":
            return k
    return None


def _compute_objective(value, objective_mode):
    if value is None:
        return None
    return float(value)


def _extract_objective_from_metrics(objective, objective_mode, val_loss_mean, val_acc_mean):
    obj = objective.strip()
    if obj == "val_loss":
        return _compute_objective(val_loss_mean, objective_mode)
    if obj in ("val_acc", "val_accuracy", "val_Accuracy"):
        return _compute_objective(val_acc_mean, objective_mode)
    if "acc" in obj.lower():
        return _compute_objective(val_acc_mean, objective_mode)
    return _compute_objective(val_loss_mean, objective_mode)


def summarize_rows(rows, objective_mode):
    """Aggregate per-config across seeds."""
    by_cfg = {}
    for r in rows:
        by_cfg.setdefault(r["config_id"], []).append(r)

    summary = []
    for cfg_id, rr in by_cfg.items():
        def collect(key):
            out = []
            for x in rr:
                v = _safe_float(x.get(key))
                if v is not None:
                    out.append(v)
            return out

        def mean_std(xs):
            if len(xs) == 0:
                return None, None
            return float(np.mean(xs)), float(np.std(xs))

        score_mean, score_std = mean_std(collect("score_mean"))
        val_loss_mean, val_loss_std = mean_std(collect("val_loss_mean"))
        val_acc_mean, val_acc_std = mean_std(collect("val_acc_mean"))
        best_epoch_mean, best_epoch_std = mean_std(collect("best_epoch_mean"))

        summary.append({
            "config_id": cfg_id,
            "n_runs": len(rr),
            "score_mean": score_mean,
            "score_std": score_std,
            "val_loss_mean": val_loss_mean,
            "val_loss_std": val_loss_std,
            "val_acc_mean": val_acc_mean,
            "val_acc_std": val_acc_std,
            "best_epoch_mean": best_epoch_mean,
            "best_epoch_std": best_epoch_std,
            "config_json": rr[0]["config_json"],
        })

    def sort_key(x):
        v = x.get("score_mean")
        if v is None:
            return np.inf if objective_mode == "min" else -np.inf
        return v

    summary.sort(key=sort_key, reverse=(objective_mode == "max"))
    return summary


def _run_holdout(run_cfg, seed, build_model_fn, load_full_data_fn, verbose):
    """Returns: (val_metrics, best_epoch, train_time_sec)"""
    X, y = load_full_data_fn(run_cfg)

    model = build_model_fn(run_cfg, seed=seed)
    trainer = Trainer(model, verbose=0)

    training_cfg = run_cfg.get("training", {})
    callbacks_cfg = run_cfg.get("callbacks", {})
    split_cfg = run_cfg.get("split", {})
    data_cfg = run_cfg.get("data", {}) # <--- LEGGI DATA CONFIG

    epochs = int(training_cfg.get("epochs", 200))
    batch_size = int(training_cfg.get("batch_size", 32))
    shuffle = bool(training_cfg.get("shuffle", True))
    include_reg_in_val = bool(training_cfg.get("include_reg_in_val", False))
    
    # Parametro normalizzazione
    normalize_data = bool(data_cfg.get("normalize", False)) # <--- NUOVO
    normalize_target = bool(data_cfg.get("normalize_target", False))

    val_size = float(split_cfg.get("val_size", 0.2))
    stratified = bool(split_cfg.get("stratified", True))

    t0 = time.time()
    _, val_metrics, history = holdout_validation(
        X=X,
        y=y,
        model=model,
        trainer=trainer,
        val_split=val_size,
        stratified=stratified,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        verbose=0,
        include_reg_in_val=include_reg_in_val,
        normalize_data=normalize_data, # <--- PASSA IL PARAMETRO
        normalize_target=normalize_target
    )
    t1 = time.time()

    best_epoch = None
    h = history.to_dict()
    if "val_loss" in h and len(h["val_loss"]) > 0:
        best_epoch = int(np.argmin(np.array(h["val_loss"], dtype=float))) + 1

    return val_metrics, best_epoch, float(t1 - t0)


def _run_kfold(run_cfg, seed, build_model_fn, load_full_data_fn, verbose):
    """Returns: (val_stats, best_epoch_mean, best_epoch_std, train_time_sec)"""
    X, y = load_full_data_fn(run_cfg)

    model = build_model_fn(run_cfg, seed=seed)
    trainer = Trainer(model, verbose=0)

    training_cfg = run_cfg.get("training", {})
    cv_cfg = run_cfg.get("cv", {})
    callbacks_cfg = run_cfg.get("callbacks", {})
    data_cfg = run_cfg.get("data", {}) # <--- LEGGI DATA CONFIG

    epochs = int(training_cfg.get("epochs", 200))
    batch_size = int(training_cfg.get("batch_size", 32))
    shuffle = bool(training_cfg.get("shuffle", True))
    include_reg_in_val = bool(training_cfg.get("include_reg_in_val", False))
    
    # Parametro normalizzazione
    normalize_data = bool(data_cfg.get("normalize", False)) # <--- NUOVO
    normalize_target = bool(data_cfg.get("normalize_target", False))

    k = int(cv_cfg.get("k", 5))
    shuffle_cv = bool(cv_cfg.get("shuffle", True))
    cv_seed = int(cv_cfg.get("seed", seed))

    t0 = time.time()
    _, val_stats, histories, _ = kfold_cross_validation(
        X=X,
        y=y,
        model=model,
        trainer=trainer,
        k=k,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=shuffle_cv,
        seed=cv_seed,
        verbose=0,
        include_reg_in_val=include_reg_in_val,
        normalize_data=normalize_data, # <--- PASSA IL PARAMETRO
        normalize_target=normalize_target
    )
    t1 = time.time()

    best_epochs = []
    for hist in histories:
        h = hist.to_dict()
        if "val_loss" in h and len(h["val_loss"]) > 0:
            be = int(np.argmin(np.array(h["val_loss"], dtype=float))) + 1
            best_epochs.append(float(be))

    if len(best_epochs) > 0:
        best_epoch_mean = float(np.mean(best_epochs))
        best_epoch_std = float(np.std(best_epochs))
    else:
        best_epoch_mean, best_epoch_std = None, None

    return val_stats, best_epoch_mean, best_epoch_std, float(t1 - t0)


def run_grid(
    config_path,
    build_model_fn,
    load_full_data_fn,
    out_csv_path,
    mode="holdout",
    seeds=None,
    objective="val_loss",
    objective_mode="auto",
    verbose=0,
):
    """Run a grid search in either holdout or k-fold mode."""
    cfg = load_config(config_path)
    run_cfgs = expand_grid(cfg)

    if seeds is None:
        base_seed = cfg.get("base", {}).get("seed", 0)
        seeds = [int(base_seed)]

    if objective_mode == "auto":
        objective_mode = _infer_mode_from_monitor(objective)
    if objective_mode not in ("min", "max"):
        raise ValueError("objective_mode must be 'min', 'max', or 'auto'.")

    mode = mode.lower().strip()
    if mode not in ("holdout", "kfold"):
        raise ValueError("mode must be 'holdout' or 'kfold'.")

    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)

    header = [
        "config_id",
        "seed",
        "mode",
        "objective",
        "objective_mode",
        "score_mean",
        "score_std",
        "val_loss_mean",
        "val_loss_std",
        "val_acc_mean",
        "val_acc_std",
        "best_epoch_mean",
        "best_epoch_std",
        "train_time_sec",
        "config_json",
    ]
    write_header = not os.path.exists(out_csv_path)

    rows = []

    total = len(run_cfgs) * len(seeds)
    done = 0

    with open(out_csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()

        for run_cfg in run_cfgs:
            config_id = _hash_config(run_cfg)

            for seed in seeds:
                done += 1
                if verbose:
                    print(f"[grid:{mode}] {done}/{total} config_id={config_id} seed={seed}")

                np.random.seed(int(seed))

                if mode == "holdout":
                    val_metrics, best_epoch, tsec = _run_holdout(
                        run_cfg, seed, build_model_fn, load_full_data_fn, verbose=verbose
                    )

                    val_loss_mean = _safe_float(val_metrics.get("loss"))
                    acc_key = _get_val_acc_key(val_metrics)
                    val_acc_mean = _safe_float(val_metrics.get(acc_key)) if acc_key else None

                    val_loss_std = 0.0 if val_loss_mean is not None else None
                    val_acc_std = 0.0 if val_acc_mean is not None else None

                    best_epoch_mean = float(best_epoch) if best_epoch is not None else None
                    best_epoch_std = 0.0 if best_epoch_mean is not None else None

                else:
                    val_stats, best_epoch_mean, best_epoch_std, tsec = _run_kfold(
                        run_cfg, seed, build_model_fn, load_full_data_fn, verbose=verbose
                    )

                    loss_stat = val_stats.get("loss", {})
                    val_loss_mean = _safe_float(loss_stat.get("mean"))
                    val_loss_std = _safe_float(loss_stat.get("std"))

                    acc_name = "Accuracy" if "Accuracy" in val_stats else None
                    if acc_name is None:
                        for k in val_stats.keys():
                            if k != "loss":
                                acc_name = k
                                break

                    if acc_name is not None:
                        acc_stat = val_stats.get(acc_name, {})
                        val_acc_mean = _safe_float(acc_stat.get("mean"))
                        val_acc_std = _safe_float(acc_stat.get("std"))
                    else:
                        val_acc_mean = None
                        val_acc_std = None

                score_mean = _extract_objective_from_metrics(
                    objective=objective,
                    objective_mode=objective_mode,
                    val_loss_mean=val_loss_mean,
                    val_acc_mean=val_acc_mean,
                )
                score_std = 0.0 if mode == "holdout" and score_mean is not None else None

                row = {
                    "config_id": config_id,
                    "seed": int(seed),
                    "mode": mode,
                    "objective": objective,
                    "objective_mode": objective_mode,
                    "score_mean": score_mean,
                    "score_std": score_std,
                    "val_loss_mean": val_loss_mean,
                    "val_loss_std": val_loss_std,
                    "val_acc_mean": val_acc_mean,
                    "val_acc_std": val_acc_std,
                    "best_epoch_mean": best_epoch_mean,
                    "best_epoch_std": best_epoch_std,
                    "train_time_sec": float(tsec),
                    "config_json": json.dumps(run_cfg, sort_keys=True),
                }

                writer.writerow(row)
                rows.append(row)

    summary = summarize_rows(rows, objective_mode=objective_mode)
    best_cfg = None
    best_score = None
    if len(summary) > 0:
        best_cfg = json.loads(summary[0]["config_json"])
        best_score = summary[0]["score_mean"]

    return {
        "best_config": best_cfg,
        "best_score": best_score,
        "rows": rows,
        "summary": summary,
    }