# training/gridsearch.py
import csv
import hashlib
import json
import os
import time
import concurrent.futures
import multiprocessing

import numpy as np

from .config import expand_grid, load_config
from .trainer import Trainer
from .holdout_cv import holdout_validation
from .kfold_cv import kfold_cross_validation


# ==================== Config & Hashing ====================

def _hash_config(cfg):
    """Generate short hash for config identification."""
    s = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def _safe_float(x):
    """Convert to float, return None if not possible."""
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _infer_mode_from_monitor(monitor):
    """Infer optimization direction from monitor name."""
    m = monitor.lower()
    if "acc" in m or "accuracy" in m:
        return "max"
    return "min"


# ==================== Metric Extraction ====================

def _get_val_acc_key(metrics_dict):
    """Extract accuracy metric name from validation metrics dict."""
    if "Accuracy" in metrics_dict:
        return "Accuracy"
    for k in metrics_dict.keys():
        if k != "loss":
            return k
    return None


def _extract_objective(objective, val_loss_mean, val_acc_mean):
    """
    Extract objective value from validation metrics.
    
    Args:
        objective: objective name (e.g., 'val_loss', 'val_acc')
        val_loss_mean: mean validation loss
        val_acc_mean: mean validation accuracy
    
    Returns:
        float or None
    """
    obj = objective.strip().lower()
    
    # Loss-based objectives
    if "loss" in obj:
        return _safe_float(val_loss_mean)
    
    # Accuracy-based objectives
    if "acc" in obj:
        return _safe_float(val_acc_mean)
    
    # Default to loss
    return _safe_float(val_loss_mean)


# ==================== Aggregation ====================

def _collect_values(rows, key):
    """Collect non-None float values for a key from list of row dicts."""
    values = []
    for row in rows:
        v = _safe_float(row.get(key))
        if v is not None:
            values.append(v)
    return values


def _compute_mean_std(values):
    """Compute mean and std from list of values."""
    if len(values) == 0:
        return None, None
    return float(np.mean(values)), float(np.std(values))


def summarize_rows(rows, objective_mode):
    """
    Aggregate per-config results across seeds.
    
    Args:
        rows: list of result dicts (one per config+seed combination)
        objective_mode: 'min' or 'max' for sorting
    
    Returns:
        list of summary dicts (one per unique config), sorted by score
    """
    # Group rows by config ID
    by_config = {}
    for row in rows:
        config_id = row["config_id"]
        by_config.setdefault(config_id, []).append(row)
    
    # Aggregate metrics for each config
    summary = []
    for config_id, config_rows in by_config.items():
        # Collect values across seeds
        score_mean, score_std = _compute_mean_std(_collect_values(config_rows, "score_mean"))
        val_loss_mean, val_loss_std = _compute_mean_std(_collect_values(config_rows, "val_loss_mean"))
        val_acc_mean, val_acc_std = _compute_mean_std(_collect_values(config_rows, "val_acc_mean"))
        best_epoch_mean, best_epoch_std = _compute_mean_std(_collect_values(config_rows, "best_epoch_mean"))
        
        summary.append({
            "config_id": config_id,
            "n_runs": len(config_rows),
            "score_mean": score_mean,
            "score_std": score_std,
            "val_loss_mean": val_loss_mean,
            "val_loss_std": val_loss_std,
            "val_acc_mean": val_acc_mean,
            "val_acc_std": val_acc_std,
            "best_epoch_mean": best_epoch_mean,
            "best_epoch_std": best_epoch_std,
            "config_json": config_rows[0]["config_json"],
        })
    
    # Sort by score (handle None values)
    def sort_key(x):
        v = x.get("score_mean")
        if v is None:
            return np.inf if objective_mode == "min" else -np.inf
        return v
    
    summary.sort(key=sort_key, reverse=(objective_mode == "max"))
    return summary


# ==================== Config Parsing ====================

def _parse_training_config(run_cfg):
    """
    Extract training hyperparameters from run config.
    
    Returns:
        dict with keys: epochs, batch_size, shuffle, include_reg_in_val,
                        normalize_data, normalize_target, val_size, stratified, k, cv_seed
    """
    training_cfg = run_cfg.get("training", {})
    split_cfg = run_cfg.get("split", {})
    cv_cfg = run_cfg.get("cv", {})
    data_cfg = run_cfg.get("data", {})
    
    return {
        # Training params
        "epochs": int(training_cfg.get("epochs", 200)),
        "batch_size": int(training_cfg.get("batch_size", 32)),
        "shuffle": bool(training_cfg.get("shuffle", True)),
        "include_reg_in_val": bool(training_cfg.get("include_reg_in_val", False)),
        
        # Normalization params
        "normalize_data": bool(data_cfg.get("normalize", False)),
        "normalize_target": bool(data_cfg.get("normalize_target", False)),
        
        # Holdout split params
        "val_size": float(split_cfg.get("val_size", 0.2)),
        "stratified": bool(split_cfg.get("stratified", True)),
        
        # K-Fold CV params
        "k": int(cv_cfg.get("k", 5)),
        "cv_seed": int(cv_cfg.get("seed", 0)),
    }


# ==================== Run Execution ====================

def _run_holdout(run_cfg, seed, build_model_fn, load_full_data_fn, verbose):
    """
    Run holdout validation for a single config+seed.
    
    Returns:
        tuple: (val_metrics, best_epoch, train_time_sec)
    """
    X, y = load_full_data_fn(run_cfg)
    model = build_model_fn(run_cfg, seed=seed)
    trainer = Trainer(model, verbose=0)
    
    # Parse config
    cfg = _parse_training_config(run_cfg)
    
    # Run validation
    t0 = time.time()
    _, val_metrics, history = holdout_validation(
        X=X,
        y=y,
        model=model,
        trainer=trainer,
        val_split=cfg["val_size"],
        stratified=cfg["stratified"],
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        shuffle=cfg["shuffle"],
        seed=seed,
        verbose=0,
        include_reg_in_val=cfg["include_reg_in_val"],
        normalize_data=cfg["normalize_data"],
        normalize_target=cfg["normalize_target"]
    )
    t1 = time.time()
    
    # Find best epoch
    best_epoch = None
    h = history.to_dict()
    if "val_loss" in h and len(h["val_loss"]) > 0:
        best_epoch = int(np.argmin(np.array(h["val_loss"], dtype=float))) + 1
    
    return val_metrics, best_epoch, float(t1 - t0)


def _run_kfold(run_cfg, seed, build_model_fn, load_full_data_fn, verbose):
    """
    Run k-fold cross-validation for a single config+seed.
    
    Returns:
        tuple: (val_stats, best_epoch_mean, best_epoch_std, train_time_sec)
    """
    X, y = load_full_data_fn(run_cfg)
    model = build_model_fn(run_cfg, seed=seed)
    trainer = Trainer(model, verbose=0)
    
    # Parse config
    cfg = _parse_training_config(run_cfg)
    
    # Run k-fold CV
    t0 = time.time()
    _, val_stats, histories, _ = kfold_cross_validation(
        X=X,
        y=y,
        model=model,
        trainer=trainer,
        k=cfg["k"],
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        shuffle=cfg["shuffle"],
        seed=cfg["cv_seed"] if cfg["cv_seed"] != 0 else seed,
        verbose=0,
        include_reg_in_val=cfg["include_reg_in_val"],
        normalize_data=cfg["normalize_data"],
        normalize_target=cfg["normalize_target"]
    )
    t1 = time.time()
    
    # Compute best epochs across folds
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


def process_single_run(args):
    """
    Execute a single config+seed run (worker function for parallel processing).
    
    Args:
        args: tuple of (run_cfg, seed, mode, build_model_fn, load_full_data_fn,
                       verbose, objective, objective_mode)
    
    Returns:
        dict: result row ready to be written to CSV
    """
    (run_cfg, seed, mode, build_model_fn, load_full_data_fn,
     verbose, objective, objective_mode) = args
    
    # Set seed for this process
    np.random.seed(int(seed))
    config_id = _hash_config(run_cfg)
    
    # Execute holdout or k-fold
    if mode == "holdout":
        val_metrics, best_epoch, train_time = _run_holdout(
            run_cfg, seed, build_model_fn, load_full_data_fn, verbose
        )
        
        val_loss_mean = _safe_float(val_metrics.get("loss"))
        acc_key = _get_val_acc_key(val_metrics)
        val_acc_mean = _safe_float(val_metrics.get(acc_key)) if acc_key else None
        
        val_loss_std = 0.0 if val_loss_mean is not None else None
        val_acc_std = 0.0 if val_acc_mean is not None else None
        
        best_epoch_mean = float(best_epoch) if best_epoch is not None else None
        best_epoch_std = 0.0 if best_epoch_mean is not None else None
    
    else:  # kfold
        val_stats, best_epoch_mean, best_epoch_std, train_time = _run_kfold(
            run_cfg, seed, build_model_fn, load_full_data_fn, verbose
        )
        
        # Extract loss stats
        loss_stat = val_stats.get("loss", {})
        val_loss_mean = _safe_float(loss_stat.get("mean"))
        val_loss_std = _safe_float(loss_stat.get("std"))
        
        # Extract accuracy stats
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
    
    # Compute objective score
    score_mean = _extract_objective(objective, val_loss_mean, val_acc_mean)
    score_std = 0.0 if mode == "holdout" and score_mean is not None else None
    
    # Build result row
    return {
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
        "train_time_sec": float(train_time),
        "config_json": json.dumps(run_cfg, sort_keys=True),
    }


# ==================== Main Grid Search ====================

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
    n_jobs=-1
):
    """
    Run parallel grid search over hyperparameter configurations.
    
    Args:
        config_path: path to config file with 'base' and 'grid' sections
        build_model_fn: function(run_cfg, seed) -> Model
        load_full_data_fn: function(run_cfg) -> (X, y)
        out_csv_path: path to save detailed results CSV
        mode: 'holdout' or 'kfold'
        seeds: list of random seeds (default: use config seed)
        objective: metric to optimize ('val_loss', 'val_acc', etc.)
        objective_mode: 'min', 'max', or 'auto'
        verbose: verbosity level
        n_jobs: number of parallel processes (-1 = all CPU cores)
    
    Returns:
        dict with keys: 'best_config', 'best_score', 'rows', 'summary'
    """
    # Load and expand grid
    cfg = load_config(config_path)
    run_cfgs = expand_grid(cfg)
    
    if seeds is None:
        base_seed = cfg.get("base", {}).get("seed", 0)
        seeds = [int(base_seed)]
    
    if objective_mode == "auto":
        objective_mode = _infer_mode_from_monitor(objective)
    
    mode = mode.lower().strip()
    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
    
    # CSV header
    header = [
        "config_id", "seed", "mode", "objective", "objective_mode",
        "score_mean", "score_std", "val_loss_mean", "val_loss_std",
        "val_acc_mean", "val_acc_std", "best_epoch_mean", "best_epoch_std",
        "train_time_sec", "config_json"
    ]
    write_header = not os.path.exists(out_csv_path)
    
    # Prepare task list
    tasks = []
    for run_cfg in run_cfgs:
        for seed in seeds:
            task_args = (
                run_cfg, seed, mode, build_model_fn,
                load_full_data_fn, verbose, objective, objective_mode
            )
            tasks.append(task_args)
    
    total = len(tasks)
    done = 0
    
    # Determine number of workers
    max_workers = multiprocessing.cpu_count() if n_jobs < 1 else n_jobs
    
    print(f"Starting Grid Search with {max_workers} processes for {total} tasks...")
    
    # Execute in parallel
    rows = []
    with open(out_csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(process_single_run, t) for t in tasks]
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    row = future.result()
                    
                    # Write result (thread-safe since in main process)
                    writer.writerow(row)
                    f.flush()
                    rows.append(row)
                    
                    done += 1
                    if verbose:
                        print(f"[grid:{mode}] {done}/{total} Config completed.")
                
                except Exception as e:
                    print(f"Error during grid search run: {e}")
    
    # Aggregate results
    summary = summarize_rows(rows, objective_mode=objective_mode)
    
    best_config = None
    best_score = None
    if len(summary) > 0:
        best_config = json.loads(summary[0]["config_json"])
        best_score = summary[0]["score_mean"]
    
    return {
        "best_config": best_config,
        "best_score": best_score,
        "rows": rows,
        "summary": summary,
    }
