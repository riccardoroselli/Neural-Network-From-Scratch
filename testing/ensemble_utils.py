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
    # NEW:
    train_color="tab:blue",
    test_color="tab:orange",
    stop_line_color="red",
    stop_line_style="--",
    stop_line_width=2.2,
    best_epoch=None,          # if None -> computed from mean TEST curve
    best_mode="auto",         # "auto" | "min" | "max"
    show_best_text=False,      # add text label near the line
):
    """
    ONE figure (full-screen-ish): faint individual runs + bold mean curves
    + red dashed vertical line at best epoch (stopping point).

    Best epoch selection (if best_epoch is None):
      - for loss:  argmin(mean test curve)
      - for acc :  argmax(mean test curve)
      - auto infers from ylabel/title (loss -> min, acc -> max)
    """
    if M_train is None or M_test is None:
        print(f"Skip plot '{title}': missing data.")
        return

    T = M_train.shape[1]
    x = np.arange(T)

    fig = plt.figure(figsize=figsize, dpi=dpi)

    # --- faint individual runs (fixed colors) ---
    for i in range(M_train.shape[0]):
        plt.plot(x, M_train[i], color=train_color, alpha=0.20)
    for i in range(M_test.shape[0]):
        plt.plot(x, M_test[i], color=test_color, alpha=0.20)

    # --- mean curves (bold, fixed colors) ---
    mean_train = np.nanmean(M_train, axis=0)
    mean_test  = np.nanmean(M_test,  axis=0)

    plt.plot(x, mean_train, color=train_color, linewidth=3.0, label="train (mean)")
    plt.plot(x, mean_test,  color=test_color,  linewidth=3.0, label="test (mean)")

    # --- choose best epoch if not provided ---
    mode = best_mode
    if mode == "auto":
        ylow = (ylabel or "").lower()
        tlow = (title or "").lower()
        if "loss" in ylow or "loss" in tlow:
            mode = "min"
        elif "acc" in ylow or "accuracy" in ylow or "acc" in tlow or "accuracy" in tlow:
            mode = "max"
        else:
            mode = "min"  # safe default

    if best_epoch is None:
        if mode == "min":
            best_epoch = int(np.nanargmin(mean_test))
        else:
            best_epoch = int(np.nanargmax(mean_test))

    # --- stop line (red dashed) ---
    plt.axvline(
        best_epoch,
        color=stop_line_color,
        linestyle=stop_line_style,
        linewidth=stop_line_width,
        label=f"best epoch = {best_epoch}",
    )

    # optional annotation text near top
    if show_best_text:
        y_top = np.nanmax(np.concatenate([mean_train[~np.isnan(mean_train)], mean_test[~np.isnan(mean_test)]]))
        plt.text(
            best_epoch,
            y_top,
            f"  stop@{best_epoch}",
            color=stop_line_color,
            rotation=90,
            va="top",
            ha="left",
        )

    # --- cosmetics ---
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

