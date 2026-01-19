# training/config.py
from __future__ import annotations

import copy
import itertools
import json
from typing import Any, Dict, List, Tuple


def load_config(path: str) -> Dict[str, Any]:
    """
    Load a configuration file.

    Supports YAML (preferred) if PyYAML is installed.
    Falls back to JSON if the file ends with .json or if YAML isn't available.

    Returns
    -------
    dict
    """
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # YAML
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise ImportError(
            "PyYAML is required to load .yaml configs. "
            "Install it (pip install pyyaml) or use .json configs."
        ) from e

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config root must be a mapping/dict, got {type(cfg)}")
    return cfg


def _is_grid_leaf(x: Any) -> bool:
    """
    A "grid leaf" is a value we want to expand, typically a list/tuple.
    We treat plain lists as grid candidates UNLESS the list represents
    a layer list (handled separately by design: layers live under model.layers).
    """
    return isinstance(x, (list, tuple))


def expand_grid(cfg: Dict[str, Any], grid_key: str = "grid") -> List[Dict[str, Any]]:
    """
    Expand a config containing a `grid:` section into a list of run configs.

    Expected structure:
      base: {... fixed params ...}
      grid:
        training.batch_size: [16, 32]
        optim.lr: [0.1, 0.05]
        model.dropout: [0.0, 0.2]

    Returns
    -------
    list[dict]
      Each element is a deep-copied config with one value chosen for each grid entry.
    """
    if "base" not in cfg:
        raise ValueError("Config must contain a top-level 'base' section.")
    base = cfg["base"]
    if not isinstance(base, dict):
        raise ValueError("'base' must be a dict.")

    grid = cfg.get(grid_key, {})
    if grid is None:
        grid = {}
    if not isinstance(grid, dict):
        raise ValueError(f"'{grid_key}' must be a dict mapping 'path.to.key' -> [values].")

    # If no grid section, return single config
    if len(grid) == 0:
        return [copy.deepcopy(base)]

    # Prepare product
    keys: List[str] = []
    values_list: List[List[Any]] = []
    for k, v in grid.items():
        if not _is_grid_leaf(v):
            raise ValueError(f"Grid value for {k!r} must be a list/tuple, got {type(v)}")
        keys.append(k)
        values_list.append(list(v))

    run_cfgs: List[Dict[str, Any]] = []
    for combo in itertools.product(*values_list):
        run_cfg = copy.deepcopy(base)
        for k, chosen in zip(keys, combo):
            set_by_path(run_cfg, k, chosen)
        run_cfgs.append(run_cfg)

    return run_cfgs


def set_by_path(d: Dict[str, Any], path: str, value: Any) -> None:
    """
    Set d['a']['b']['c'] = value given path 'a.b.c'.

    Creates intermediate dicts if needed.
    """
    parts = path.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def get_by_path(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    Get d['a']['b']['c'] given path 'a.b.c'. Returns default if not found.
    """
    parts = path.split(".")
    cur: Any = d
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur
