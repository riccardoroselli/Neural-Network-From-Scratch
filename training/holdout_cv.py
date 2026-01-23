# training/holdout_cv.py
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
    normalize_data=False,
    normalize_target=False,
    **fit_kwargs
):
    """
    Perform holdout validation with proper train/validation split.
    
    Prevents data leakage by computing normalization statistics exclusively
    on the training set and applying them to the validation set.
    
    Workflow:
        1. Split data into train/validation
        2. Normalize train set (fit statistics)
        3. Normalize validation set (using train statistics)
        4. Train model
        5. Evaluate on both train and validation sets
    
    Args:
        X: input features, shape (N, D)
        y: target values, shape (N, ...) 
        model: Model instance to train
        trainer: Trainer instance
        val_split: fraction of data for validation (default: 0.2)
        stratified: whether to use stratified split (for classification)
        epochs: number of training epochs
        batch_size: mini-batch size
        shuffle: whether to shuffle before split
        seed: random seed for reproducibility
        verbose: verbosity level (0=silent, 1=progress, 2=detailed)
        include_reg_in_val: whether to include regularization in validation loss
        normalize_data: if True, normalize X using train statistics
        normalize_target: if True, normalize y using train statistics
        **fit_kwargs: additional arguments passed to trainer.fit()
    
    Returns:
        tuple: (train_metrics, val_metrics, history)
            - train_metrics: dict of metrics on training set
            - val_metrics: dict of metrics on validation set
            - history: History object with per-epoch logs
    
    Note:
        If normalize_targets=True, returned metrics are in normalized scale.
        To get original scale metrics, denormalize predictions before evaluation.
    
    Example:
        >>> train_metrics, val_metrics, history = holdout_validation(
        ...     X, y, model, trainer,
        ...     val_split=0.2,
        ...     normalize_inputs=True,
        ...     normalize_targets=True,
        ...     epochs=100
        ... )
    """
    # Prepare stratification
    stratify_param = None
    if stratified:
        stratify_param = y
        if not shuffle:
            if verbose >= 2:
                print("[Holdout] Stratification requires shuffle=True, enabling shuffle")
            shuffle = True
    
    # 1. Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, 
        y, 
        test_size=val_split, 
        shuffle=shuffle, 
        random_state=seed, 
        stratify=stratify_param
    )
    
    if verbose >= 1:
        print(f"\n{'='*60}")
        print(f"Holdout Validation (split: {int((1-val_split)*100)}/{int(val_split*100)})")
        print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")
        print(f"{'='*60}")
    
    # 2. Normalize inputs (fit on train, transform on val)
    if normalize_data:
        X_train, mean_X, std_X = normalize(X_train)
        X_val, _, _ = normalize(X_val, mean=mean_X, std=std_X)
        
        if verbose >= 2:
            print(f"[Normalized X] Train mean={mean_X.mean():.4f}, std={std_X.mean():.4f}")
    
    # 3. Normalize targets (fit on train, transform on val)
    if normalize_target:
        y_train, mean_y, std_y = normalize(y_train)
        y_val, _, _ = normalize(y_val, mean=mean_y, std=std_y)
        
        if verbose >= 2:
            print(f"[Normalized y] Train mean={mean_y.mean():.4f}, std={std_y.mean():.4f}")
    
    # 4. Reset model to initial state
    if verbose >= 2:
        print("[Holdout] Resetting model weights")
    model.reset()
    
    # 5. Train model
    history = trainer.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,  # Always shuffle during training
        seed=seed,
        include_reg_in_val=include_reg_in_val,
        **fit_kwargs
    )
    
    # 6. Evaluate on both sets
    train_metrics = trainer.evaluate(
        X_train, y_train, 
        batch_size=batch_size,
        include_regularization=include_reg_in_val
    )
    val_metrics = trainer.evaluate(
        X_val, y_val, 
        batch_size=batch_size,
        include_regularization=include_reg_in_val
    )
    
    if verbose >= 1:
        print(f"\n{'='*60}")
        print("Final Metrics:")
        print(f"  Train: {_format_metrics(train_metrics)}")
        print(f"  Val:   {_format_metrics(val_metrics)}")
        print(f"{'='*60}\n")
    
    return train_metrics, val_metrics, history


def _format_metrics(metrics):
    """Helper to format metrics dict for display"""
    return ', '.join([f'{k}={v:.4f}' for k, v in metrics.items()])
