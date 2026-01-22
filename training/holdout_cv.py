import numpy as np
from sklearn.model_selection import train_test_split
from data_handler.data_loader import normalize

def holdout_validation(
    X,
    y,
    model,
    trainer,
    val_split=0.2,
    stratified=False,
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
    Esegue Holdout Validation con opzione per normalizzare X e y.
    Evita il data leakage calcolando le statistiche solo sul Train.
    """
    
    # Gestione stratificazione
    stratify_param = y if stratified else None

    # 1. SPLIT
    X_train, X_val, y_train, y_val = train_test_split(
        X, 
        y, 
        test_size=val_split, 
        shuffle=shuffle, 
        random_state=seed, 
        stratify=stratify_param
    )

    if normalize_data:
        X_train, mean_x, std_x = normalize(X_train)
        X_val, _, _ = normalize(X_val, mean=mean_x, std=std_x)

    if normalize_target:
        y_train, mean_y, std_y = normalize(y_train)
        y_val, _, _ = normalize(y_val, mean=mean_y, std=std_y)

    if verbose >= 1:
        norm_flags = []
        if normalize_data:
            norm_flags.append("X normalized")
        if normalize_target:
            norm_flags.append("y normalized")
        norm_str = f" ({', '.join(norm_flags)})" if norm_flags else ""
        print(f"\n{'─'*60}")
        print(f"Holdout Validation{norm_str}")
        print(f"{'─'*60}")
        print(f"  Train: {len(X_train)} samples | Val: {len(X_val)} samples")
        print(f"{'─'*60}")


    # Reset pesi modello
    model.reset()

    # 4. TRAIN
    history = trainer.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        include_reg_in_val=include_reg_in_val,
        **fit_kwargs
    )

    # 5. EVALUATE (Nota: le metriche saranno nella scala normalizzata se normalize_target=True)
    train_metrics = trainer.evaluate(X_train, y_train, batch_size=batch_size)
    val_metrics = trainer.evaluate(X_val, y_val, batch_size=batch_size)

    return train_metrics, val_metrics, history