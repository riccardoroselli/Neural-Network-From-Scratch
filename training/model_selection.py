# training/model_selection.py
import json
import os

from .config import expand_grid, load_config
from .gridsearch import run_grid, summarize_rows
from .refine_grid import refine_grid_from_topk


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _write_csv(path, rows):
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
):
    """Two-phase model selection: coarse (holdout) over full grid, then fine (kfold) over top-K."""
    os.makedirs(out_dir, exist_ok=True)

    coarse_csv = os.path.join(out_dir, "coarse_runs.csv")
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
    )
    coarse_summary = coarse_out["summary"]
    _write_csv(os.path.join(out_dir, "coarse_summary.csv"), coarse_summary)

    if top_k <= 0:
        top_k = len(coarse_summary)
    top_cfgs = [json.loads(r["config_json"]) for r in coarse_summary[:top_k]]

    cfg0 = load_config(config_path)
    coarse_grid = cfg0.get("grid", {}) or {}

    sel = cfg0.get("selection", {}) or {}
    refine = sel.get("refine", {}) or {}

    top_k_cfg = sel.get("top_k", None)
    if top_k_cfg is not None:
        top_k = int(top_k_cfg)

    steps = int(refine.get("steps", 5))
    sig_digits = int(refine.get("sig_digits", 1))
    rules = refine.get("rules", {}) or {}

    fine_grid, refine_report = refine_grid_from_topk(
        coarse_grid=coarse_grid,
        topk_cfgs=top_cfgs,
        rules=rules,
        steps=steps,
        sig_digits=sig_digits,
    )

    _write_json(os.path.join(out_dir, "fine_grid.json"), fine_grid)
    _write_json(os.path.join(out_dir, "refine_report.json"), refine_report)

    fine_cfg_path = os.path.join(out_dir, "fine_config.json")
    _write_json(fine_cfg_path, {"base": cfg0.get("base", {}), "grid": fine_grid})

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
    )

    fine_summary = fine_out["summary"]
    _write_csv(os.path.join(out_dir, "fine_summary.csv"), fine_summary)

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