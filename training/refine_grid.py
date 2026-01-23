# training/refine_grid.py
import math
import numpy as np


def round_sig(x, sig=1):
    """Round x to `sig` significant digits."""
    if x == 0.0:
        return 0.0
    if not math.isfinite(x):
        return float(x)
    return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)


def _get_by_path(d, path):
    cur = d
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            raise KeyError(path)
        cur = cur[p]
    return cur


def compute_minmax_from_topk(topk_cfgs, param_path):
    """Extract scalar values for param_path from each config in topk_cfgs and return (min, max)."""
    vals = []
    for cfg in topk_cfgs:
        try:
            v = _get_by_path(cfg, param_path)
        except KeyError:
            continue
        vals.append(float(v))

    if len(vals) == 0:
        raise KeyError(f"Refinable key '{param_path}' not found in any top-K config.")
    return float(min(vals)), float(max(vals))


def build_linear_values(low, high, steps=5, sig_digits=1, clip=None):
    """Create `steps` linearly spaced values in [low, high], rounded and clipped."""
    steps = int(steps)
    if steps <= 1:
        steps = 1

    if low == high or steps == 1:
        v = round_sig(float(low), sig=sig_digits)
        if clip is not None:
            v = min(max(v, clip[0]), clip[1])
        return [v]

    arr = np.linspace(float(low), float(high), num=steps, dtype=float)

    out = []
    for v in arr.tolist():
        rv = round_sig(float(v), sig=sig_digits)
        if clip is not None:
            rv = min(max(rv, clip[0]), clip[1])
        out.append(rv)

    out = sorted(set(out))
    return out


def refine_grid_from_topk(coarse_grid, topk_cfgs, rules, steps=5, sig_digits=1):
    """Build a refined fine grid from coarse grid + topK configs using linear spacing and rounding."""
    fine_grid = dict(coarse_grid)
    report = {"refined": {}}

    for param_path, rule in (rules or {}).items():
        low, high = compute_minmax_from_topk(topk_cfgs, param_path)

        clip = None
        if isinstance(rule, dict) and "clip" in rule and rule["clip"] is not None:
            c = rule["clip"]
            clip = (float(c[0]), float(c[1]))

        values = build_linear_values(
            low=low,
            high=high,
            steps=steps,
            sig_digits=sig_digits,
            clip=clip,
        )

        fine_grid[param_path] = values

        report["refined"][param_path] = {
            "min_topk": low,
            "max_topk": high,
            "steps": int(steps),
            "sig_digits": int(sig_digits),
            "clip": list(clip) if clip is not None else None,
            "values": values,
        }

    return fine_grid, report