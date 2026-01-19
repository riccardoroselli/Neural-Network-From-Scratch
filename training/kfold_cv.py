# nn/kfold.py

import numpy as np
from sklearn.model_selection import StratifiedKFold


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
    **fit_kwargs
):
    
   # StratifiedKFold per split bilanciati
    skf = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=seed)

    # Flatten y se necessario per StratifiedKFold
    y_flat = y.ravel() if y.ndim > 1 else y


    fold_results = []
    histories = []

    print(f"\nStarting {k}-Fold Cross Validation")
    print("=" * 60)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y_flat)):
        if verbose >= 1:
            print(f"\n{'='*60}")
            print(f"Fold {fold_idx + 1}/{k}")
            print('='*60)

        # Split con indici
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if verbose >= 2:
            print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")

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
            shuffle=shuffle,
            seed=seed + fold_idx if seed is not None else None,
            **fit_kwargs
        )

        # Evaluate on train and val
        train_metrics = trainer.evaluate(X_train, y_train, batch_size=batch_size)
        val_metrics = trainer.evaluate(X_val, y_val, batch_size=batch_size)

        fold_results.append({
            "fold": fold_idx + 1,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics
        })
        histories.append(history)

        if verbose >= 1:
            print(f"\nFold {fold_idx + 1} Results:")
            for name, val in train_metrics.items():
                print(f"  Train {name}: {val:.4f}")
            for name, val in val_metrics.items():
                print(f"  Val {name}: {val:.4f}")


    # Compute statistics across folds
    train_stats = _compute_fold_statistics(fold_results, "train_metrics")
    val_stats = _compute_fold_statistics(fold_results, "val_metrics")

    # Print summary
    print(f"\n{'='*60}")
    print(f"K-Fold Cross Validation Summary ({k} folds)")
    print('='*60)
    print("\nTraining Metrics (mean ± std):")
    for metric_name, stat in train_stats.items():
        print(f"  {metric_name:12s}: {stat['mean']:.4f} ± {stat['std']:.4f}")
    print("\nValidation Metrics (mean ± std):")
    for metric_name, stat in val_stats.items():
        print(f"  {metric_name:12s}: {stat['mean']:.4f} ± {stat['std']:.4f}")

    print('='*60)

    return train_stats, val_stats, histories, fold_results


def _compute_fold_statistics(fold_results, key):
    """
    Compute mean and std of metrics across folds.

    Parameters
    ----------
    fold_results : list
        List of fold results
    key : str
        Key to extract from each fold ("train_metrics" or "val_metrics")

    Returns
    -------
    dict : {"metric_name": {"mean": ..., "std": ...}, ...}
    """
    # Collect all metrics
    all_metrics = {}
    for fold in fold_results:
        metrics = fold[key]
        for metric_name, value in metrics.items():
            if metric_name not in all_metrics:
                all_metrics[metric_name] = []
            all_metrics[metric_name].append(float(value))

    # Compute statistics
    stats = {}
    for metric_name, values in all_metrics.items():
        stats[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "values": values
        }

    return stats

