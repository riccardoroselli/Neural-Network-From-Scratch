# experiments/CUP/common.py
import os
import numpy as np
from data_handler.data_loader import load_cup
from training.model_factory import build_model_from_cfg
from training.model_selection import run_two_phase_selection

def load_full_data(run_cfg):
    # RITORNA I DATI GREZZI.
    # GridSearch -> Holdout/KFold applicheranno normalize() se richiesto dal config.
    data_cfg = run_cfg.get("data", {})
    train_path = data_cfg["train_path"]
    X, y = load_cup(train_path, training=True)
    return X, y

def build_model(run_cfg, seed):
    # CUP: 12 Input, 4 Output, Regression
    return build_model_from_cfg(
        run_cfg, 
        seed=seed, 
        in_dim=12, 
        out_dim=4, 
        task="regression"
    )

def create_internal_split(original_path, train_save_path, test_save_path, test_ratio=0.20, seed=42):
    print(f"[CUP] Loading raw: {original_path}")
    X, y = load_cup(original_path, training=True)
    data = np.hstack([X, y])
    
    np.random.seed(seed)
    np.random.shuffle(data)
    
    test_size = int(len(data) * test_ratio)
    train_data = data[test_size:]
    test_data = data[:test_size]
    
    # Save with dummy ID for load_cup compatibility
    def save_dummy(fname, d):
        ids = np.arange(len(d)).reshape(-1, 1)
        d_out = np.hstack([ids, d])
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        np.savetxt(fname, d_out, delimiter=',', fmt='%f')
        
    save_dummy(train_save_path, train_data)
    save_dummy(test_save_path, test_data)
    print(f"[CUP] Saved split: {train_save_path}, {test_save_path}")

def run_cup_selection(config_path, out_dir, seeds, top_k=5):
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
    )