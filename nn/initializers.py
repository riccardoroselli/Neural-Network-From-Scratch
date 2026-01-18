import numpy as np

def xavier_uniform(in_dim, out_dim, rng):
    """
    Xavier/Glorot uniform initialization.
    Good for layers with tanh or sigmoid activations.
    """
    limit = np.sqrt(6.0 / (in_dim + out_dim))
    return rng.uniform(-limit, limit, size=(in_dim, out_dim))


def he_uniform(in_dim, out_dim, rng):
    """
    He uniform initialization.
    Good for layers with ReLU activations.
    """
    limit = np.sqrt(6.0 / in_dim)
    return rng.uniform(-limit, limit, size=(in_dim, out_dim))


def zeros(in_dim, out_dim, rng):
    """Zero initialization (not recommended for weights, OK for biases)"""
    return np.zeros((in_dim, out_dim))

