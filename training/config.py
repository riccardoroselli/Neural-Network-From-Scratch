# training/config.py
import copy
import itertools
import json


def load_config(path):
    """Load a configuration file (JSON or YAML)."""
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    try:
        import yaml
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


def _is_grid_leaf(x):
    """Check if value is a grid candidate (list or tuple)."""
    return isinstance(x, (list, tuple))


def expand_grid(cfg, grid_key="grid"):
    """Expand a config containing a grid section into a list of run configs."""
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

    if len(grid) == 0:
        return [copy.deepcopy(base)]

    keys = []
    values_list = []
    for k, v in grid.items():
        if not _is_grid_leaf(v):
            raise ValueError(f"Grid value for {k!r} must be a list/tuple, got {type(v)}")
        keys.append(k)
        values_list.append(list(v))

    run_cfgs = []
    for combo in itertools.product(*values_list):
        run_cfg = copy.deepcopy(base)
        for k, chosen in zip(keys, combo):
            set_by_path(run_cfg, k, chosen)
        run_cfgs.append(run_cfg)

    return run_cfgs


def set_by_path(d, path, value):
    """Set d['a']['b']['c'] = value given path 'a.b.c'."""
    parts = path.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def get_by_path(d, path, default=None):
    """Get d['a']['b']['c'] given path 'a.b.c'. Returns default if not found."""
    parts = path.split(".")
    cur = d
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur