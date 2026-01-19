# nn/kfold.py

import numpy as np


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
    
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # Shuffle data
    if shuffle:
        indices = rng.permutation(len(X))
        X = X[indices]
        y = y[indices]

    # Split into k folds
    X_folds = np.array_split(X, k)
    y_folds = np.array_split(y, k)

    fold_results = []
    histories = []

    print(f"\nStarting {k}-Fold Cross Validation")
    print("=" * 60)

    for fold_idx in range(k):
        if verbose >= 1:
            print(f"\n{'='*60}")
            print(f"Fold {fold_idx + 1}/{k}")
            print('='*60)

        # Create train/val split
        X_val = X_folds[fold_idx]
        y_val = y_folds[fold_idx]

        X_train = np.concatenate([X_folds[i] for i in range(k) if i != fold_idx])
        y_train = np.concatenate([y_folds[i] for i in range(k) if i != fold_idx])

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
            print(f"  Train: {_format_metrics(train_metrics)}")
            print(f"  Val:   {_format_metrics(val_metrics)}")

    # Compute statistics across folds
    train_stats = _compute_fold_statistics(fold_results, "train_metrics")
    val_stats = _compute_fold_statistics(fold_results, "val_metrics")

    # Print summary
    print(f"\n{'='*60}")
    print(f"K-Fold Cross Validation Summary ({k} folds)")
    print('='*60)
    print("\nTraining Metrics (mean ± std):")
    _print_statistics(train_stats)
    print("\nValidation Metrics (mean ± std):")
    _print_statistics(val_stats)
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


def _format_metrics(metrics_dict):
    """Format metrics dictionary as string."""
    return ", ".join([f"{k}={v:.4f}" for k, v in metrics_dict.items()])


def _print_statistics(stats_dict):
    """Print statistics in a readable format."""
    for metric_name, stat in stats_dict.items():
        mean = stat["mean"]
        std = stat["std"]
        print(f"  {metric_name:12s}: {mean:.4f} ± {std:.4f}")
