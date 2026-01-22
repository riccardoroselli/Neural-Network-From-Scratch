import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# Histories utilities
# -------------------------
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


# -------------------------
# Ensemble utilities
# -------------------------
def predict_proba(model, X, batch_size=256):
    """Calls model.forward(X, training=False), returns probs shape (N,)."""
    probs = []
    for i in range(0, len(X), batch_size):
        Xb = X[i:i + batch_size]
        yb = model.forward(Xb, training=False)
        probs.append(np.asarray(yb).reshape(-1))
    return np.concatenate(probs, axis=0)

def majority_vote(P, threshold=0.5):
    """
    P: (M, N) probabilities
    Returns: (y_hat, p_mean, vote_frac)
    """
    M = P.shape[0]
    hard = (P >= threshold).astype(int)
    votes = hard.sum(axis=0)
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

def mean_std(vals):
    vals = np.asarray(vals, dtype=float)
    return float(np.nanmean(vals)), float(np.nanstd(vals))


# -------------------------
# Plotting (TWO separate figures)
# -------------------------
def plot_runs_with_mean(
    M_train, M_test, title, ylabel,
    ylim=None,
    figsize=(14, 6),
    dpi=130,
    margins=(0.055, 0.99, 0.90, 0.14),  # left, right, top, bottom
):
    """
    ONE figure (full-screen-ish): faint individual runs + bold mean curves.
    """
    if M_train is None or M_test is None:
        print(f"Skip plot '{title}': missing data.")
        return

    T = M_train.shape[1]
    x = np.arange(T)

    fig = plt.figure(figsize=figsize, dpi=dpi)

    for i in range(M_train.shape[0]):
        plt.plot(x, M_train[i], alpha=0.20)
    for i in range(M_test.shape[0]):
        plt.plot(x, M_test[i], alpha=0.20)

    plt.plot(x, np.nanmean(M_train, axis=0), linewidth=3.0, label="train (mean)")
    plt.plot(x, np.nanmean(M_test,  axis=0), linewidth=3.0, label="test (mean)")

    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.legend()

    left, right, top, bottom = margins
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)

    plt.show()
