import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from data_handler.data_loader import normalize

def kfold_cross_validation(
    X,
    y,
    model,
    trainer,
    k=5,
    epochs=100,
    batch_size=32,
    shuffle=True,
    seed=None,
    verbose=1,
    include_reg_in_val=False,
    normalize_data=False,    # Normalizzazione Input (X)
    normalize_target=False,  # Normalizzazione Target (y)
    **fit_kwargs
):
    """
    Esegue K-Fold Cross Validation con normalizzazione dinamica per fold.
    """

    # --- RILEVAMENTO TIPO TASK (Regressione vs Classificazione) ---
    is_regression = False
    
    # 1. Controllo tipo dati
    try:
        if np.issubdtype(y.dtype, np.floating):
            is_regression = True
    except:
        pass

    # 2. Controllo dimensioni (Multi-output = Regressione, es. CUP)
    # StratifiedKFold rompe se y ha shape (N, 4)
    if y.ndim > 1 and y.shape[1] > 1:
        is_regression = True

    # --- SELEZIONE SPLITTER ---
    if is_regression:
        if verbose >= 1: 
            print(f"Task: Regression/Multi-output detected (y shape: {y.shape}) -> Using Standard KFold")
        # KFold accetta y multi-dimensionale senza problemi
        cv = KFold(n_splits=k, shuffle=shuffle, random_state=seed)
        y_split = y 
    else:
        if verbose >= 1:
            print(f"Task: Classification detected -> Using StratifiedKFold")
        cv = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=seed)
        # Stratified richiede y 1D
        y_split = y.ravel() if y.ndim > 1 else y

    fold_results = []
    histories = []

    print(f"\nStarting {k}-Fold Cross Validation")

    # Ciclo sui Fold
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y_split)):
        if verbose >= 1:
            print(f"Fold {fold_idx + 1}/{k}")

        # Slice dei dati grezzi
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # --- NORMALIZZAZIONE INPUT (X) ---
        if normalize_data:
            # Calcolo statistiche su questo specifico training fold
            X_train, mean_x, std_x = normalize(X_train)
            # Applico le stesse statistiche al validation fold
            X_val, _, _ = normalize(X_val, mean=mean_x, std=std_x)

        # --- NORMALIZZAZIONE TARGET (y) ---
        if normalize_target:
            # Calcolo statistiche su questo specifico training fold
            y_train, mean_y, std_y = normalize(y_train)
            # Applico le stesse statistiche al validation fold
            # Fondamentale per avere una Loss coerente (MSE su dati scalati)
            y_val, _, _ = normalize(y_val, mean=mean_y, std=std_y)

        # Reset Modello
        model.reset()

        # Training
        history = trainer.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed + fold_idx if seed is not None else None,
            include_reg_in_val=include_reg_in_val,
            **fit_kwargs
        )

        # Valutazione
        train_metrics = trainer.evaluate(X_train, y_train, batch_size=batch_size)
        val_metrics = trainer.evaluate(X_val, y_val, batch_size=batch_size)

        fold_results.append({
            "fold": fold_idx + 1,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics
        })
        histories.append(history)

    # Helper per aggregare le statistiche
    def _compute_stats(results, key):
        stats = {}
        if not results: return stats
        metric_names = results[0][key].keys()
        for m in metric_names:
            vals = [r[key][m] for r in results]
            stats[m] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals))
            }
        return stats

    train_stats = _compute_stats(fold_results, "train_metrics")
    val_stats = _compute_stats(fold_results, "val_metrics")

    return train_stats, val_stats, histories, fold_results