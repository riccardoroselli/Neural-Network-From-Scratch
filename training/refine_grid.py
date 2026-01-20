# training/refine_grid.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import math
import numpy as np


def round_sig(x: float, sig: int = 1) -> float:
    """
    Round x to `sig` significant digits (default: 1 significant digit).
    Examples:
      0.083 -> 0.08  (sig=1 => 0.08? actually 0.08 has 1 sig digit? It's 8e-2)
      0.012 -> 0.01
      123.0 -> 100.0
    """
    if x == 0.0:
        return 0.0
    if not math.isfinite(x):
        return float(x)
    return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)


def _get_by_path(d: Dict[str, Any], path: str) -> Any:
    cur: Any = d
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            raise KeyError(path)
        cur = cur[p]
    return cur


def compute_minmax_from_topk(
    topk_cfgs: List[Dict[str, Any]],
    param_path: str,
) -> Tuple[float, float]:
    """
    Extract scalar values for param_path from each config in topk_cfgs and return (min, max).
    """
    vals: List[float] = []
    for cfg in topk_cfgs:
        try:
            v = _get_by_path(cfg, param_path)
        except KeyError:
            continue
        # refinable keys must be scalar numeric
        vals.append(float(v))

    if len(vals) == 0:
        raise KeyError(f"Refinable key '{param_path}' not found in any top-K config.")
    return float(min(vals)), float(max(vals))


def build_linear_values(
    low: float,
    high: float,
    steps: int = 5,
    sig_digits: int = 1,
    clip: Tuple[float, float] | None = None,
) -> List[float]:
    """
    Create `steps` linearly spaced values in [low, high], rounded to `sig_digits`
    significant digits, clipped if requested, then unique-sorted.
    """
    steps = int(steps)
    if steps <= 1:
        steps = 1

    if low == high or steps == 1:
        v = round_sig(float(low), sig=sig_digits)
        if clip is not None:
            v = min(max(v, clip[0]), clip[1])
        return [v]

    arr = np.linspace(float(low), float(high), num=steps, dtype=float)

    out: List[float] = []
    for v in arr.tolist():
        rv = round_sig(float(v), sig=sig_digits)
        if clip is not None:
            rv = min(max(rv, clip[0]), clip[1])
        out.append(rv)

    # unique + sorted
    out = sorted(set(out))
    return out


def refine_grid_from_topk(
    coarse_grid: Dict[str, Any],
    topk_cfgs: List[Dict[str, Any]],
    rules: Dict[str, Any],
    steps: int = 5,
    sig_digits: int = 1,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Build a refined fine grid from coarse grid + topK configs.

    SPECIFICATIONS (as requested):
      - refinable keys explicitly come from `rules` mapping
      - bounds are computed with MIN/MAX over top-K (NO quantiles)
      - NO expansion (no 0.8/1.2 multipliers)
      - values are generated with LINEAR spacing
      - default steps=5
      - values rounded to ONE significant digit (sig_digits=1 by default)
      - clipping rules supported via rules[key]["clip"] = [low, high]
      - categorical parameters kept EXACTLY as in coarse grid (Approach B):
          fine grid starts as a copy of coarse grid, then overrides only refinable keys

    Returns:
      fine_grid: dict path->list
      report: dict describing bounds and generated values
    """
    # Approach B: keep all coarse grid entries by default
    fine_grid: Dict[str, Any] = dict(coarse_grid)

    report: Dict[str, Any] = {"refined": {}}

    # Override only refinable keys
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
