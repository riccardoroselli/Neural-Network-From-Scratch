# training/refine_grid.py
import math
import numpy as np


def round_sig(x, sig=1):
    """
    Round x to `sig` significant digits.
    
    Examples:
        round_sig(0.12345, 2) -> 0.12
        round_sig(123.45, 3) -> 123.0
        round_sig(0.0, 1) -> 0.0
    """
    if x == 0.0:
        return 0.0
    if not math.isfinite(x):
        return float(x)
    
    exponent = int(math.floor(math.log10(abs(x))))
    decimals = sig - exponent - 1
    return round(x, decimals)


def _get_by_path(config, path):
    """
    Access nested dict value using dot notation.
    
    Args:
        config: nested dict
        path: dot-separated path (e.g., 'optim.lr')
    
    Returns:
        value at path
    
    Raises:
        KeyError: if path not found
    
    Example:
        _get_by_path({'optim': {'lr': 0.01}}, 'optim.lr') -> 0.01
    """
    current = config
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            raise KeyError(path)
        current = current[key]
    return current


def compute_minmax_from_topk(topk_cfgs, param_path):
    """
    Extract scalar values for param_path from top-K configs and compute range.
    
    Args:
        topk_cfgs: list of config dicts (best configs from coarse search)
        param_path: dot-separated path (e.g., 'optim.lr')
    
    Returns:
        tuple: (min_value, max_value)
    
    Raises:
        KeyError: if param_path not found in any config
    
    Example:
        configs = [{'optim': {'lr': 0.01}}, {'optim': {'lr': 0.1}}]
        compute_minmax_from_topk(configs, 'optim.lr') -> (0.01, 0.1)
    """
    values = []
    
    for cfg in topk_cfgs:
        try:
            value = _get_by_path(cfg, param_path)
            values.append(float(value))
        except KeyError:
            continue
    
    if len(values) == 0:
        raise KeyError(f"Refinable key '{param_path}' not found in any top-K config.")
    
    return float(min(values)), float(max(values))


def build_linear_values(low, high, steps=5, sig_digits=1, clip=None):
    """
    Create linearly spaced values in [low, high], rounded and optionally clipped.
    
    Args:
        low: lower bound
        high: upper bound
        steps: number of values to generate
        sig_digits: significant digits for rounding
        clip: optional tuple (min_clip, max_clip) for hard bounds
    
    Returns:
        list of unique sorted values
    
    Example:
        build_linear_values(0.01, 0.1, steps=5, sig_digits=2)
        -> [0.01, 0.032, 0.055, 0.078, 0.1]
    """
    steps = max(1, int(steps))
    
    # Handle degenerate case: single value
    if low == high or steps == 1:
        value = round_sig(low, sig=sig_digits)
        if clip is not None:
            value = _apply_clip(value, clip)
        return [value]
    
    # Generate linearly spaced values
    raw_values = np.linspace(low, high, num=steps, dtype=float)
    
    # Round and clip each value
    processed_values = []
    for value in raw_values:
        rounded = round_sig(value, sig=sig_digits)
        if clip is not None:
            rounded = _apply_clip(rounded, clip)
        processed_values.append(rounded)
    
    # Remove duplicates and sort
    unique_values = sorted(set(processed_values))
    return unique_values


def refine_grid_from_topk(coarse_grid, topk_cfgs, rules, steps=5, sig_digits=1):
    """
    Build a refined fine grid from coarse grid + top-K configs.
    
    For each parameter in `rules`, computes min/max from top-K configs,
    then generates linearly spaced values with optional clipping.
    
    Args:
        coarse_grid: original grid dict (unchanged parameters kept as-is)
        topk_cfgs: list of best config dicts from coarse search
        rules: dict mapping param_path -> rule (None or dict with 'clip')
        steps: number of values to generate per parameter
        sig_digits: significant digits for rounding
    
    Returns:
        tuple: (fine_grid dict, report dict with refinement details)
    
    Example:
        coarse_grid = {'optim.lr': [0.001, 0.01, 0.1]}
        topk_cfgs = [{'optim': {'lr': 0.01}}, {'optim': {'lr': 0.1}}]
        rules = {'optim.lr': {'clip': (0.005, 0.5)}}
        fine_grid, report = refine_grid_from_topk(coarse_grid, topk_cfgs, rules, steps=5)
    """
    fine_grid = dict(coarse_grid)
    report = {"refined": {}}
    
    for param_path, rule in (rules or {}).items():
        # Compute range from top-K configs
        min_value, max_value = compute_minmax_from_topk(topk_cfgs, param_path)
        
        # Extract clip bounds if provided
        clip_bounds = _parse_clip_from_rule(rule)
        
        # Generate refined values
        values = build_linear_values(
            low=min_value,
            high=max_value,
            steps=steps,
            sig_digits=sig_digits,
            clip=clip_bounds,
        )
        
        # Update grid and report
        fine_grid[param_path] = values
        
        report["refined"][param_path] = {
            "min_topk": min_value,
            "max_topk": max_value,
            "steps": steps,
            "sig_digits": sig_digits,
            "clip": list(clip_bounds) if clip_bounds is not None else None,
            "values": values,
        }
    
    return fine_grid, report


# ==================== Private Helpers ====================

def _apply_clip(value, clip_bounds):
    """
    Clip value to [min_bound, max_bound].
    
    Args:
        value: scalar to clip
        clip_bounds: tuple (min_bound, max_bound)
    
    Returns:
        clipped value
    """
    min_bound, max_bound = clip_bounds
    return max(min_bound, min(value, max_bound))


def _parse_clip_from_rule(rule):
    """
    Extract clip bounds from rule dict.
    
    Args:
        rule: None or dict with optional 'clip' key
    
    Returns:
        tuple (min_clip, max_clip) or None
    """
    if not isinstance(rule, dict):
        return None
    
    clip = rule.get("clip")
    if clip is None:
        return None
    
    return (float(clip[0]), float(clip[1]))
