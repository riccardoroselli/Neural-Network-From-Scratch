# testing/ensemble_utils.py

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Basic stats
# ------------------------------------------------------------
def mean_std(vals):
    vals = np.asarray(vals, dtype=float)
    return float(np.nanmean(vals)), float(np.nanstd(vals))


# ------------------------------------------------------------
# Histories stacking (NaN pad, robust to early stopping lengths)
# ------------------------------------------------------------
def _as_float_array(x):
    if x is None:
        return None
    return np.asarray(x, dtype=float)

def stack_histories(histories, key):
    """Return (num_runs, max_len) padded with NaN."""
    series = [_as_float_array(h.get(key, None)) for h in histories]
    series = [s for s in series if s is not None]
    if len(series) == 0:
        return None
    max_len = max(len(s) for s in series)
    M = np.full((len(series), max_len), np.nan, dtype=float)
    for i, s in enumerate(series):
        M[i, :len(s)] = s
    return M


# ------------------------------------------------------------
# Ensemble prediction + majority voting
# ------------------------------------------------------------
def predict_proba(model, X, batch_size=256):
    """
    Framework-compatible probability prediction:
    uses model.forward(X, training=False), returns (N,)
    """
    probs = []
    for i in range(0, len(X), batch_size):
        Xb = X[i:i + batch_size]
        yb = model.forward(Xb, training=False)
        probs.append(np.asarray(yb).reshape(-1))
    return np.concatenate(probs, axis=0)

def majority_vote(P, threshold=0.5):
    """
    P: (M, N) probabilities
    Returns:
      y_hat: (N,) hard-voted labels
      p_mean: (N,) mean probability
      vote_frac: (N,) fraction of models voting 1
    """
    M = P.shape[0]
    hard = (P >= threshold).astype(int)  # (M, N)
    votes = hard.sum(axis=0)            # (N,)
    p_mean = P.mean(axis=0)

    y_hat = (votes > (M / 2)).astype(int)
    ties = (votes == (M / 2))
    if np.any(ties):
        y_hat[ties] = (p_mean[ties] >= threshold).astype(int)

    return y_hat, p_mean, votes / M

def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_pred = np.asarray(y_pred).reshape(-1).astype(int)
    return float(np.mean(y_true == y_pred))

def bce_loss(y_true, p, eps=1e-12):
    y_true = np.asarray(y_true).reshape(-1).astype(float)
    p = np.asarray(p).reshape(-1).astype(float)
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


# ------------------------------------------------------------
# Best-epoch from K-Fold CV on TRAIN ONLY (NO TEST LEAK)
# ------------------------------------------------------------
def compute_best_epoch_from_kfold(
    X_train,
    y_train,
    cfg,
    seed,
    build_model_from_cfg_fn,
    trainer_cls,
    k=5,
    monitor="val_loss",
):
    """
    Runs k-fold CV on (X_train, y_train) only, and returns:
      best_epoch_mean (float), best_epoch_std (float), best_epochs_list (list[int])

    Notes:
    - Uses the validation split inside k-fold (NOT the external test set).
    - Epoch indexing returned is 0-based (so it aligns with plotting indices).
    """
    from training.kfold_cv import kfold_cross_validation  # local import to avoid circular deps

    # training params
    training_cfg = cfg.get("training", {}) or {}
    epochs = int(training_cfg.get("epochs", 1000))
    batch_size = int(training_cfg.get("training_batch_size", training_cfg.get("batch_size", 16)))
    include_reg_in_val = bool(training_cfg.get("include_reg_in_val", False))

    # some configs store cv params
    cv_cfg = cfg.get("cv", {}) or {}
    k = int(cv_cfg.get("k", k))

    # build ONE model + trainer; kfold_cross_validation will call model.reset() every fold
    np.random.seed(int(seed))
    model = build_model_from_cfg_fn(
        run_cfg=cfg,
        seed=int(seed),
        in_dim=X_train.shape[1],
        out_dim=y_train.shape[1],
        task="binary",
    )
    trainer = trainer_cls(model, verbose=0)

    # run k-fold (returns histories per fold)
    _, _, histories, _ = kfold_cross_validation(
        X=X_train,
        y=y_train,
        model=model,
        trainer=trainer,
        k=k,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        seed=int(seed),
        verbose=0,
        include_reg_in_val=include_reg_in_val,
    )

    # extract best epoch per fold from fold histories
    best_epochs = []
    for hist in histories:
        h = hist.to_dict()

        if monitor not in h or len(h[monitor]) == 0:
            continue

        arr = np.asarray(h[monitor], dtype=float)

        # choose min for loss, max for accuracy-like
        if "acc" in monitor.lower() or "accuracy" in monitor.lower():
            be = int(np.nanargmax(arr))   # 0-based
        else:
            be = int(np.nanargmin(arr))   # 0-based

        best_epochs.append(be)

    if len(best_epochs) == 0:
        return None, None, []

    return float(np.mean(best_epochs)), float(np.std(best_epochs)), best_epochs


# ------------------------------------------------------------
# Plotting (FULL SCREEN-ish, BLUE train / ORANGE test, red stop line)
# ------------------------------------------------------------
def plot_runs_with_mean(
    M_train,
    M_test,
    title,
    ylabel,
    ylim=None,
    figsize=(14, 6),
    dpi=130,
    margins=(0.055, 0.99, 0.90, 0.14),  # left, right, top, bottom
    best_epoch=None,                    # 0-based
    best_label="best epoch",
):
    """
    ONE figure: faint individual runs + bold mean curves.
    - Train curves: blue
    - Test curves : orange
    - Optional vertical red dashed line at best_epoch (0-based)
    """
    if M_train is None or M_test is None:
        print(f"Skip plot '{title}': missing data.")
        return

    T = max(M_train.shape[1], M_test.shape[1])
    x = np.arange(T)

    fig = plt.figure(figsize=figsize, dpi=dpi)

    # faint runs
    for i in range(M_train.shape[0]):
        plt.plot(np.arange(M_train.shape[1]), M_train[i], alpha=0.15, color="tab:blue")
    for i in range(M_test.shape[0]):
        plt.plot(np.arange(M_test.shape[1]),  M_test[i],  alpha=0.15, color="tab:orange")

    # mean (bold)
    plt.plot(np.arange(M_train.shape[1]), np.nanmean(M_train, axis=0), linestyle="--", linewidth=3.0, color="tab:blue",   label="train (mean)")
    plt.plot(np.arange(M_test.shape[1]),  np.nanmean(M_test,  axis=0), linewidth=3.0, color="tab:orange", label="test (mean)")

    # best epoch vertical line
    if best_epoch is not None:
        be = int(best_epoch)
        plt.axvline(be, color="red", linestyle="--", linewidth=2.5, label=f"{best_label} = {be}")

    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.35)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.legend()

    left, right, top, bottom = margins
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)

    plt.show()
