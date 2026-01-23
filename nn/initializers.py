# nn/initializers.py
import numpy as np


def xavier_uniform(in_dim, out_dim, rng):
    """
    Xavier/Glorot uniform initialization.
    
    Samples weights from uniform distribution U(-limit, limit) where:
        limit = sqrt(6 / (in_dim + out_dim))
    
    Best for:
        - Tanh activations
        - Sigmoid activations
    
    Maintains variance across layers to prevent vanishing/exploding gradients.
    
    Args:
        in_dim: number of input neurons
        out_dim: number of output neurons
        rng: numpy random generator
    
    Returns:
        weight matrix of shape (in_dim, out_dim)
    """
    limit = np.sqrt(6.0 / (in_dim + out_dim))
    return rng.uniform(-limit, limit, size=(in_dim, out_dim))


def he_uniform(in_dim, out_dim, rng):
    """
    He uniform initialization.
    
    Samples weights from uniform distribution U(-limit, limit) where:
        limit = sqrt(6 / in_dim)
    
    Best for:
        - ReLU activations
        - Leaky ReLU activations
    
    Designed specifically for ReLU which kills negative activations.
    
    Args:
        in_dim: number of input neurons
        out_dim: number of output neurons
        rng: numpy random generator
    
    Returns:
        weight matrix of shape (in_dim, out_dim)
    """
    limit = np.sqrt(6.0 / in_dim)
    return rng.uniform(-limit, limit, size=(in_dim, out_dim))


def zeros(in_dim, out_dim, rng=None):
    """
    Zero initialization.
    
    Returns a matrix of zeros. Suitable for biases, but NOT for weights
    (would cause symmetry breaking problem).
    
    Args:
        in_dim: first dimension
        out_dim: second dimension
        rng: unused (kept for consistent interface)
    
    Returns:
        zero matrix of shape (in_dim, out_dim)
    """
    return np.zeros((in_dim, out_dim))
