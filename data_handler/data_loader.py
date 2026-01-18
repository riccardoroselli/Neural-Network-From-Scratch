# nn/utils.py
import numpy as np

# --- MONK DATASETS UTILS ---

def load_monk(path, encode=True):
    """
    Loads and parses a MONK dataset file.
    
    Args:
        path: path to the .train or .test file.
        encode: if True, automatically applies 1-of-k encoding (17 features).
    
    Returns:
        X: features (encoded or raw)
        y: targets (0 or 1)
    """
    data = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # Split by space 
                parts = line.split()
                data.append([int(x) for x in parts[:-1]]) #Ignore the last column (id)
    
    data = np.array(data)
    y = data[:, 0].reshape(-1, 1) # Target is column 0
    X_raw = data[:, 1:]           # Features are columns 1 to 6
    
    if not encode:
        return X_raw, y

    # Automatic 1-of-k encoding (17 units total)
    # Count of distinct values for each feature 
    attr_value_counts = [3, 3, 2, 3, 4, 2]
    X_encoded = []
    
    for row in X_raw:
        encoded_row = []
        for i, val in enumerate(row):
            # Create a zero vector for the current feature
            bits = np.zeros(attr_value_counts[i])
            # Set 1 at the correct index (values in MONK are 1-indexed)
            bits[val - 1] = 1
            encoded_row.extend(bits)
        X_encoded.append(encoded_row)
        
    return np.array(X_encoded), y

# --- CUP DATASET UTILS ---

def load_cup(path, training=True):
    """
    Parses the ML-CUP CSV files.
    Training format: id, 12 inputs, 4 targets
    Test format: id, 12 inputs

    If training is set to true there will be uploaded the input variables and the target variables, otherwise just the inputs
    """
    # Genfromtxt handles headers (starting with #) and comma delimiters
    data = np.genfromtxt(path, delimiter=',', comments='#')
    
    X = data[:, 1:13] #column 0 is the ID (discarded), columns 1-12 are the inputs
    
    if training:
        # Columns 13-16 are the targets
        y = data[:, 13:17]
        return X, y
    
    return X


def normalize(data, mean=None, std=None):
    """
    Standardizes data (zero mean, unit variance).

    """
    if mean is None:
        mean = np.mean(data, axis=0)
    if std is None:
        std = np.std(data, axis=0)
        # Avoid division by zero
        std[std == 0] = 1.0
        
    return (data - mean) / std, mean, std


def denormalize(y_norm, mean, std):
    """
    Reverts the standardization process to bring data back to its original scale.
    
    """
    return (y_norm * std) + mean