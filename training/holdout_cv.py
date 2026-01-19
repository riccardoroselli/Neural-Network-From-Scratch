# nn/holdout_sklearn.py

import numpy as np
from sklearn.model_selection import train_test_split

def holdout_validation(
    X,
    y,
    model,
    trainer,
    val_split=0.2,
    stratified=False,  # Nuovo parametro per attivare la stratificazione
    epochs=100,
    batch_size=32,
    shuffle=True,
    seed=None,
    verbose=1,
    **fit_kwargs
):
    """
    Esegue una validazione Holdout utilizzando train_test_split di scikit-learn.
    Supporta la stratificazione per problemi di classificazione.
    """
    
    # Gestione del parametro stratify
    # Se stratified=True, usiamo le etichette y per bilanciare il split.
    # Nota: stratify richiede shuffle=True.
    stratify_param = y if stratified else None
    
    if stratified and not shuffle:
        print("Warning: 'stratified' è True ma 'shuffle' è False. La stratificazione richiede lo shuffle. Forzo shuffle=True.")
        shuffle = True

    # Utilizzo di train_test_split di scikit-learn
    X_train, X_val, y_train, y_val = train_test_split(
        X, 
        y, 
        test_size=val_split, 
        shuffle=shuffle, 
        random_state=seed, 
        stratify=stratify_param
    )

    if verbose >= 1:
        print(f"\nStarting Holdout Validation (Sklearn)")
        print("=" * 60)
        print(f"Split params: val_split={val_split}, stratified={stratified}, seed={seed}")
        print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")
        
        # Se stratificato o classificazione, potrebbe essere utile stampare la distribuzione delle classi
        if stratified or (len(y.shape) == 1 or y.shape[1] == 1):
             # Esempio basilare per stampare distribuzione classi (assumendo y scalari o one-hot decodificati)
             pass 

    # Reset model weights
    model.reset()

    # Train
    history = trainer.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True, # Shuffle dei batch durante il training (indipendente dallo split)
        seed=seed,
        **fit_kwargs
    )

    # Evaluate on train and val
    train_metrics = trainer.evaluate(X_train, y_train, batch_size=batch_size)
    val_metrics = trainer.evaluate(X_val, y_val, batch_size=batch_size)

    if verbose >= 1:
        print(f"\nResults:")
        print(f"  Train: {train_metrics}")
        print(f"  Val: {val_metrics}")
        print('='*60)

    return train_metrics, val_metrics, history