# training/model_selection.py
import json
import os

from .config import load_config
from .gridsearch import run_grid
from .refine_grid import refine_grid_from_topk


def _write_json(path, obj):
    """Write JSON to file, creating directories if needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _write_csv(path, rows):
    """Write list of dicts to CSV, creating directories if needed."""
    import csv
    
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    
    if len(rows) == 0:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    
    header = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)


def run_two_phase_selection(
    config_path,
    build_model_fn,
    load_full_data_fn,
    out_dir,
    seeds,
    coarse_objective="val_loss",
    coarse_objective_mode="auto",
    coarse_mode="holdout",
    top_k=10,
    fine_objective="val_loss",
    fine_objective_mode="auto",
    fine_mode="kfold",
    verbose=1,
    n_jobs=-1,
):
    """
    Two-phase model selection strategy:
    
    Phase 1 (Coarse): Fast holdout search over full hyperparameter grid
    Phase 2 (Fine): Rigorous k-fold CV over top-K configs from Phase 1
    
    Args:
        config_path: Path to config JSON with 'base' and 'grid' sections
        build_model_fn: Function that takes run_cfg dict and returns Model
        load_full_data_fn: Function that returns (X, y) training data
        out_dir: Directory to save all outputs
        seeds: List of random seeds for reproducibility
        coarse_objective: Metric name for Phase 1 ranking (e.g., 'val_loss')
        coarse_objective_mode: 'min' or 'max' or 'auto' for coarse ranking
        coarse_mode: 'holdout' or 'kfold' for Phase 1 validation
        top_k: Number of best configs to refine in Phase 2
        fine_objective: Metric name for Phase 2 ranking
        fine_objective_mode: 'min' or 'max' or 'auto' for fine ranking
        fine_mode: 'holdout' or 'kfold' for Phase 2 validation
        verbose: Logging level (0=silent, 1=progress, 2=debug)
        n_jobs: Number of parallel workers (-1 = all CPUs)
    
    Returns:
        dict with keys:
            - best_config: Winning hyperparameter configuration
            - best_score: Final objective value
            - coarse_summary: List of all Phase 1 results
            - fine_summary: List of all Phase 2 results
            - out_dir: Output directory path
            - fine_config_path: Path to refined grid config
    
    Output Files:
        {out_dir}/coarse_runs.csv: All Phase 1 individual runs
        {out_dir}/coarse_summary.csv: Phase 1 aggregated results
        {out_dir}/fine_grid.json: Refined grid ranges
        {out_dir}/refine_report.json: Grid refinement statistics
        {out_dir}/fine_config.json: Config for Phase 2
        {out_dir}/fine_runs.csv: All Phase 2 individual runs
        {out_dir}/fine_summary.csv: Phase 2 aggregated results
        {out_dir}/best_config.json: Winner configuration
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # ========== Phase 1: Coarse Search ==========
    # run_grid appends to its output file, so a stale file from an earlier or
    # interrupted search would leave its rows mixed in with this one's. The
    # summary is built from the current run's in-memory rows and would not
    # notice, but the CSV would no longer be a faithful record of the search.
    # Phase 2 already clears its file; phase 1 must do the same.
    coarse_csv = os.path.join(out_dir, "coarse_runs.csv")
    if os.path.exists(coarse_csv):
        os.remove(coarse_csv)

    coarse_out = run_grid(
        config_path=config_path,
        build_model_fn=build_model_fn,
        load_full_data_fn=load_full_data_fn,
        out_csv_path=coarse_csv,
        mode=coarse_mode,
        seeds=seeds,
        objective=coarse_objective,
        objective_mode=coarse_objective_mode,
        verbose=verbose,
        n_jobs=n_jobs
    )
    
    coarse_summary = coarse_out["summary"]
    _write_csv(os.path.join(out_dir, "coarse_summary.csv"), coarse_summary)
    
    # Extract top-K configs for refinement
    cfg0 = load_config(config_path)
    top_k_final = _get_top_k(cfg0, top_k)
    
    if top_k_final <= 0:
        top_k_final = len(coarse_summary)
    
    top_cfgs = [json.loads(r["config_json"]) for r in coarse_summary[:top_k_final]]
    
    # ========== Grid Refinement ==========
    coarse_grid = cfg0.get("grid", {}) or {}
    refine_params = _extract_refine_params(cfg0)
    
    fine_grid, refine_report = refine_grid_from_topk(
        coarse_grid=coarse_grid,
        topk_cfgs=top_cfgs,
        rules=refine_params["rules"],
        steps=refine_params["steps"],
        sig_digits=refine_params["sig_digits"],
    )
    
    _write_json(os.path.join(out_dir, "fine_grid.json"), fine_grid)
    _write_json(os.path.join(out_dir, "refine_report.json"), refine_report)
    
    # Create fine-tuning config
    fine_cfg_path = os.path.join(out_dir, "fine_config.json")
    fine_cfg = {"base": cfg0.get("base", {}), "grid": fine_grid}
    _write_json(fine_cfg_path, fine_cfg)
    
    # ========== Phase 2: Fine Search ==========
    fine_csv = os.path.join(out_dir, "fine_runs.csv")
    if os.path.exists(fine_csv):
        os.remove(fine_csv)
    
    fine_out = run_grid(
        config_path=fine_cfg_path,
        build_model_fn=build_model_fn,
        load_full_data_fn=load_full_data_fn,
        out_csv_path=fine_csv,
        mode=fine_mode,
        seeds=seeds,
        objective=fine_objective,
        objective_mode=fine_objective_mode,
        verbose=verbose,
        n_jobs=n_jobs
    )
    
    fine_summary = fine_out["summary"]
    _write_csv(os.path.join(out_dir, "fine_summary.csv"), fine_summary)
    
    # Save best config
    best_config = fine_out.get("best_config", None)
    best_score = fine_out.get("best_score", None)
    _write_json(os.path.join(out_dir, "best_config.json"), best_config)
    
    return {
        "best_config": best_config,
        "best_score": best_score,
        "coarse_summary": coarse_summary,
        "fine_summary": fine_summary,
        "out_dir": out_dir,
        "fine_config_path": fine_cfg_path,
    }


def _get_top_k(cfg, default_top_k):
    """
    Extract top_k from config, with fallback to parameter.
    Config takes precedence to allow easy tuning via JSON.
    """
    selection = cfg.get("selection", {}) or {}
    top_k_cfg = selection.get("top_k", None)
    
    if top_k_cfg is not None:
        return int(top_k_cfg)
    
    return default_top_k


def _extract_refine_params(cfg):
    """
    Extract grid refinement parameters from config with defaults.
    
    Returns:
        dict with keys: steps, sig_digits, rules
    """
    selection = cfg.get("selection", {}) or {}
    refine = selection.get("refine", {}) or {}
    
    return {
        "steps": int(refine.get("steps", 5)),
        "sig_digits": int(refine.get("sig_digits", 1)),
        "rules": refine.get("rules", {}) or {},
    }
