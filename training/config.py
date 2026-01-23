# training/config.py
import copy
import itertools
import json


def load_config(path):
    """
    Load a configuration file (JSON or YAML).
    
    Args:
        path: path to config file (.json or .yaml/.yml)
    
    Returns:
        dict: configuration dictionary
    
    Raises:
        ValueError: if config root is not a dict
        ImportError: if PyYAML is not installed for .yaml files
    
    Example:
        cfg = load_config('configs/monk1.json')
    """
    # Load JSON
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
    
    # Load YAML
    else:
        try:
            import yaml
        except Exception as e:
            raise ImportError(
                "PyYAML is required to load .yaml configs. "
                "Install it (pip install pyyaml) or use .json configs."
            ) from e
        
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    
    # Validate root is dict
    if not isinstance(config, dict):
        raise ValueError(f"Config root must be a mapping/dict, got {type(config)}")
    
    return config


def is_grid_value(value):
    """
    Check if value is a grid candidate (list or tuple).
    
    Grid values are arrays of alternatives to try during grid search.
    Non-grid values are fixed constants.
    
    Args:
        value: any value from config
    
    Returns:
        bool: True if value is list/tuple (grid candidate)
    
    Example:
        is_grid_value([0.01, 0.1, 1.0]) -> True
        is_grid_value(0.1) -> False
    """
    return isinstance(value, (list, tuple))


def expand_grid(cfg, grid_key="grid"):
    """
    Expand a config containing a 'grid' section into a list of run configs.
    
    The config must have:
    - 'base': baseline config (all fixed values)
    - 'grid': dict mapping dot-paths to lists of values to try
    
    Args:
        cfg: config dict with 'base' and 'grid' sections
        grid_key: name of grid section (default: 'grid')
    
    Returns:
        list of run config dicts (Cartesian product of grid values)
    
    Raises:
        ValueError: if 'base' is missing or invalid
        ValueError: if grid values are not lists/tuples
    
    Example:
        cfg = {
            'base': {'optim': {'lr': 0.01}},
            'grid': {'optim.lr': [0.01, 0.1, 1.0]}
        }
        expand_grid(cfg) -> [
            {'optim': {'lr': 0.01}},
            {'optim': {'lr': 0.1}},
            {'optim': {'lr': 1.0}}
        ]
    """
    # Validate base section
    if "base" not in cfg:
        raise ValueError("Config must contain a top-level 'base' section.")
    
    base = cfg["base"]
    if not isinstance(base, dict):
        raise ValueError("'base' must be a dict.")
    
    # Get grid section (optional)
    grid = cfg.get(grid_key, {})
    if grid is None:
        grid = {}
    
    if not isinstance(grid, dict):
        raise ValueError(f"'{grid_key}' must be a dict mapping path.to.key -> values.")
    
    # If grid is empty, return single base config
    if len(grid) == 0:
        return [copy.deepcopy(base)]
    
    # Validate grid values and collect keys/values
    keys = []
    values_list = []
    
    for param_path, values in grid.items():
        if not is_grid_value(values):
            raise ValueError(
                f"Grid value for {param_path!r} must be a list/tuple, got {type(values)}"
            )
        keys.append(param_path)
        values_list.append(list(values))
    
    # Generate all combinations (Cartesian product)
    run_configs = []
    
    for combination in itertools.product(*values_list):
        run_config = copy.deepcopy(base)
        
        # Set each parameter in the combination
        for param_path, chosen_value in zip(keys, combination):
            set_by_path(run_config, param_path, chosen_value)
        
        run_configs.append(run_config)
    
    return run_configs


def set_by_path(config, path, value):
    """
    Set config['a']['b']['c'] = value given path 'a.b.c'.
    
    Creates intermediate dicts if they don't exist.
    
    Args:
        config: nested dict to modify in-place
        path: dot-separated path (e.g., 'optim.lr')
        value: value to set
    
    Example:
        cfg = {}
        set_by_path(cfg, 'optim.lr', 0.01)
        # cfg is now {'optim': {'lr': 0.01}}
    """
    keys = path.split(".")
    current = config
    
    # Navigate to parent, creating dicts as needed
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    
    # Set the final value
    current[keys[-1]] = value


def get_by_path(config, path, default=None):
    """
    Get config['a']['b']['c'] given path 'a.b.c'.
    
    Returns default if path not found.
    
    Args:
        config: nested dict
        path: dot-separated path (e.g., 'optim.lr')
        default: value to return if path not found
    
    Returns:
        value at path or default
    
    Example:
        cfg = {'optim': {'lr': 0.01}}
        get_by_path(cfg, 'optim.lr') -> 0.01
        get_by_path(cfg, 'optim.momentum', 0.0) -> 0.0
    """
    keys = path.split(".")
    current = config
    
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    
    return current
