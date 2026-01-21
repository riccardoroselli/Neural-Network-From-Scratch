# experiments/MONK/common.py
import numpy as np

from training.model_factory import build_model_from_cfg
from training.model_selection import run_two_phase_selection
from data_handler.data_loader import load_monk


def load_full_data(run_cfg):
    data_cfg = run_cfg.get("data", {})
    train_path = data_cfg["train_path"]
    X, y = load_monk(train_path, encode=True)
    return X, y


def build_model(run_cfg, seed):
    return build_model_from_cfg(run_cfg, seed=seed, in_dim=17, out_dim=1, task="binary")


def run_monk_selection(config_path, out_dir, seeds, top_k, n_jobs=-1):
    return run_two_phase_selection(
        config_path=config_path,
        build_model_fn=build_model,
        load_full_data_fn=load_full_data,
        out_dir=out_dir,
        seeds=seeds,
        coarse_mode="holdout",
        fine_mode="kfold",
        top_k=top_k,
        coarse_objective="val_loss",
        fine_objective="val_loss",
        verbose=1,
        n_jobs=n_jobs
    )