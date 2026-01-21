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
    if stratified and not shuffle:
        shuffle = True

    # 1. SPLIT
    X_train, X_val, y_train, y_val = train_test_split(
        X, 
        y, 
        test_size=val_split, 
        shuffle=shuffle, 
        random_state=seed, 
        stratify=stratify_param
    )

    # 2. NORMALIZZAZIONE INPUT (X)
    if normalize_data:
        # Fit su Train
        X_train, mean_x, std_x = normalize(X_train)
        # Transform su Val (usando statistiche train)
        X_val, _, _ = normalize(X_val, mean=mean_x, std=std_x)
        
        if verbose >= 1:
            print(f"[Holdout] X Normalized. Train Mean[0]={mean_x[0]:.3f}")

    # 3. NORMALIZZAZIONE TARGET (y)
    if normalize_target:
        # Fit su Train
        y_train, mean_y, std_y = normalize(y_train)
        # Transform su Val
        # È necessario scalare anche y_val perché il modello predirà valori scalati,
        # quindi la loss deve essere calcolata confrontando mele con mele.
        y_val, _, _ = normalize(y_val, mean=mean_y, std=std_y)
        
        if verbose >= 1:
            print(f"[Holdout] y Normalized. Train Mean[0]={mean_y[0]:.3f}")

    if verbose >= 1:
        print(f"\nStarting Holdout Validation")
        print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")

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