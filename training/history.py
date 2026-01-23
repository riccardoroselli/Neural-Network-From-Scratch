# nn/history.py

class History:
    """
    Container for per-epoch training metrics.
    
    Stores metric values as lists, allowing tracking of metrics over epochs.
    Keys typically include: 'loss', 'val_loss', 'Accuracy', 'val_Accuracy', etc.
    
    Example:
        history = History()
        history.log(loss=0.12, Accuracy=0.98)
        history.log(loss=0.10, Accuracy=0.99)
        
        print(history.last('loss'))  # 0.10
        print(history.get('loss'))   # [0.12, 0.10]
    """

    def __init__(self):
        self.logs = {}

    def log(self, **kwargs):
        """
        Record metric values for current epoch.
        
        Args:
            **kwargs: metric_name=value pairs to log
        """
        for key, value in kwargs.items():
            self.logs.setdefault(key, []).append(value)

    def last(self, key, default=None):
        """
        Get the most recent value for a metric.
        
        Args:
            key: metric name
            default: value to return if metric doesn't exist
        
        Returns:
            last logged value, or default if not found
        """
        if key not in self.logs:
            return default
        return self.logs[key][-1]

    def get(self, key, default=None):
        """
        Get the full list of values for a metric.
        
        Args:
            key: metric name
            default: value to return if metric doesn't exist
        
        Returns:
            list of all logged values, or default if not found
        """
        return self.logs.get(key, default)

    def to_dict(self):
        """
        Convert to dictionary.
        
        Returns:
            dict mapping metric names to lists of values
        """
        return self.logs

    def keys(self):
        """Get all tracked metric names"""
        return self.logs.keys()

    def __repr__(self):
        """String representation showing tracked metrics"""
        if not self.logs:
            return "History(empty)"
        
        keys = ", ".join(sorted(self.logs.keys()))
        return f"History(keys=[{keys}])"
