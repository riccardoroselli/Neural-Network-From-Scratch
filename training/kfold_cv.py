# training/kfold_cv.py
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from data.data_handler.data_loader import normalize


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
    normalize_data=False,
    normalize_target=False,
    **fit_kwargs
):
    """
    Perform K-Fold Cross Validation with per-fold normalization.
    
    Prevents data leakage by computing normalization statistics separately
    for each fold, using only the training split of that fold.
    
    Workflow (per fold):
        1. Split data into train/validation for this fold
        2. Normalize train split (fit statistics)
        3. Normalize validation split (using train statistics)
        4. Reset and train model
        5. Evaluate on both train and validation splits
        6. Aggregate statistics across all folds
    
    Args:
        X: input features, shape (N, D)
        y: target values, shape (N, ...) 
        model: Model instance to train
        trainer: Trainer instance
        k: number of folds (default: 5)
        epochs: number of training epochs per fold
        batch_size: mini-batch size
        shuffle: whether to shuffle before splitting
        seed: random seed for reproducibility
        verbose: verbosity level (0=silent, 1=progress, 2=detailed)
        include_reg_in_val: whether to include regularization in validation loss
        normalize_data: if True, normalize X per-fold using train statistics
        normalize_target: if True, normalize y per-fold using train statistics
        **fit_kwargs: additional arguments passed to trainer.fit()
    
    Returns:
        tuple: (train_stats, val_stats, histories, fold_results)
            - train_stats: dict with mean/std of train metrics across folds
            - val_stats: dict with mean/std of validation metrics across folds
            - histories: list of History objects (one per fold)
            - fold_results: list of dicts with per-fold metrics
    
    Note:
        Automatically detects task type (see _is_regression_task):
        - Multi-output targets (y.shape[1] > 1) -> Regression -> KFold
        - Floating-point targets              -> Regression -> KFold
        - Otherwise (discrete 1-D targets)    -> Classification -> StratifiedKFold
    """
    # Detect task type (classification vs regression)
    is_regression = _is_regression_task(y)
    
    # Select appropriate cross-validation splitter
    if is_regression:
        if verbose >= 1:
            print(f"{'='*60}")
            print(f"Task: Regression (y.shape={y.shape})")
            print(f"Using KFold with k={k}")
            print(f"{'='*60}")
        cv_splitter = KFold(n_splits=k, shuffle=shuffle, random_state=seed)
        y_for_split = y
    else:
        if verbose >= 1:
            print(f"{'='*60}")
            print(f"Task: Classification (y.shape={y.shape})")
            print(f"Using StratifiedKFold with k={k}")
            print(f"{'='*60}")
        cv_splitter = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=seed)
        # Stratified requires 1D targets
        y_for_split = y.ravel() if y.ndim > 1 else y
    
    # Storage for results
    fold_results = []
    histories = []
    
    # Iterate over folds
    for fold_idx, (train_indices, val_indices) in enumerate(cv_splitter.split(X, y_for_split)):
        if verbose >= 1:
            print(f"\nFold {fold_idx + 1}/{k}")
            print(f"-" * 60)
        
        # 1. Split data for this fold
        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]
        
        if verbose >= 2:
            print(f"  Train: {len(X_train)} samples, Val: {len(X_val)} samples")
        
        # 2. Normalize inputs (fit on train fold, transform on val fold)
        if normalize_data:
            X_train, mean_X, std_X = normalize(X_train)
            X_val, _, _ = normalize(X_val, mean=mean_X, std=std_X)
            
            if verbose >= 2:
                print(f"  [Normalized X] mean={mean_X.mean():.4f}, std={std_X.mean():.4f}")
        
        # 3. Normalize targets (fit on train fold, transform on val fold)
        if normalize_target:
            y_train, mean_y, std_y = normalize(y_train)
            y_val, _, _ = normalize(y_val, mean=mean_y, std=std_y)
            
            if verbose >= 2:
                print(f"  [Normalized y] mean={mean_y.mean():.4f}, std={std_y.mean():.4f}")
        
        # 4. Reset model to initial state
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
            seed=None if seed is None else seed + fold_idx,
            include_reg_in_val=include_reg_in_val,
            **fit_kwargs
        )
        
        # 6. Evaluate on both train and validation
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
        
        # Store results
        fold_results.append({
            "fold": fold_idx + 1,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics
        })
        histories.append(history)
        
        if verbose >= 1:
            print(f"  Train: {_format_metrics(train_metrics)}")
            print(f"  Val:   {_format_metrics(val_metrics)}")
    
    # 7. Aggregate statistics across folds
    train_stats = _compute_fold_statistics(fold_results, "train_metrics")
    val_stats = _compute_fold_statistics(fold_results, "val_metrics")
    
    # Print summary
    if verbose >= 1:
        print(f"\n{'='*60}")
        print(f"K-Fold Cross Validation Summary (k={k})")
        print(f"{'='*60}")
        print("Train Metrics (mean ± std):")
        for metric_name, stats in train_stats.items():
            print(f"  {metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print("\nValidation Metrics (mean ± std):")
        for metric_name, stats in val_stats.items():
            print(f"  {metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"{'='*60}\n")
    
    return train_stats, val_stats, histories, fold_results


def _is_regression_task(y):
    """
    Detect if task is regression or classification based on targets.
    
    Heuristics:
        - Multi-output (y.shape[1] > 1) → Regression
        - Floating point dtype → Regression
        - Otherwise → Classification
    
    Args:
        y: target array
    
    Returns:
        bool: True if regression, False if classification
    """
    # Multi-output always regression
    if y.ndim > 1 and y.shape[1] > 1:
        return True
    
    # Check dtype
    if np.issubdtype(y.dtype, np.floating):
        return True
    
    return False


def _compute_fold_statistics(fold_results, metrics_key):
    """
    Compute mean and std of metrics across folds.
    
    Args:
        fold_results: list of dicts with per-fold results
        metrics_key: key to extract from each fold ('train_metrics' or 'val_metrics')
    
    Returns:
        dict: {metric_name: {'mean': ..., 'std': ...}}
    """
    if not fold_results:
        return {}
    
    stats = {}
    metric_names = fold_results[0][metrics_key].keys()
    
    for metric_name in metric_names:
        values = [fold[metrics_key][metric_name] for fold in fold_results]
        stats[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values))
        }
    
    return stats


def _format_metrics(metrics):
    """Helper to format metrics dict for display"""
    return ', '.join([f'{k}={v:.4f}' for k, v in metrics.items()])
