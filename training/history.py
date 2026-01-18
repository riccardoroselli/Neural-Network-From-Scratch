class History:
    """
    Container for per-epoch training logs.

    Stores values as lists under keys:
        "loss", "Accuracy", "val_loss", "val_Accuracy", ...

    Example:
        history.log(loss=0.12, Accuracy=0.98, val_loss=0.15, val_Accuracy=0.96)
    """

    def __init__(self):
        self.logs = {}

    def log(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.logs:
                self.logs[k] = []
            self.logs[k].append(v)

    def last(self, key, default=None):
        if key not in self.logs or len(self.logs[key]) == 0:
            return default
        return self.logs[key][-1]

    def to_dict(self):
        return dict(self.logs)

    def keys(self):
        return self.logs.keys()

    def __repr__(self):
        keys = ", ".join(sorted(self.logs.keys()))
        return f"History(keys=[{keys}])"
