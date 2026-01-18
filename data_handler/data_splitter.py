import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold

def split_train_val(X, y, val_size=0.2, random_state=None, shuffle=True, stratify=None):
    """
    Divide il dataset in training set e validation set.
    
    Args:
        X (np.ndarray): Le features del dataset.
        y (np.ndarray): I target del dataset.
        val_size (float): La proporzione del dataset da includere nel validation split (es. 0.2 per il 20%).
        random_state (int, optional): Seed per il generatore di numeri casuali per riproducibilità.
        shuffle (bool): Se mescolare i dati prima dello split (default: True).
        stratify (array-like, optional): Se non None, i dati sono splittati in modo stratificato, 
                                         preservando la percentuale di campioni per ogni classe.
                                         (Consigliato per i dataset MONK, non applicabile direttamente per CUP regressione).

    Returns:
        tuple: (X_train, X_val, y_train, y_val)
    """
    
    # Esegue lo split usando scikit-learn
    X_train, X_val, y_train, y_val = train_test_split(
        X, 
        y, 
        test_size=val_size, 
        random_state=random_state, 
        shuffle=shuffle, 
        stratify=stratify
    )
    
    return X_train, X_val, y_train, y_val

def get_kfold_generator(X, y, k=5, shuffle=True, random_state=None, stratify=False):
    """
    Crea un generatore per iterare sugli indici di train e validation per la K-Fold Cross Validation.
    
    Args:
        X (np.ndarray): Dataset features.
        y (np.ndarray): Dataset targets.
        k (int): Numero di fold (es. 4, 5, 10).
        shuffle (bool): Se mescolare i dati prima di splittare.
        random_state (int): Seed per la riproducibilità.
        stratify (bool): Se True, usa StratifiedKFold (per classificazione, mantiene il bilanciamento delle classi).
                         Se False, usa KFold standard (per regressione).
                         
    Yields:
        (indices_train, indices_val): Tuple contenenti gli indici delle righe per train e validation corrente.
    """
    if stratify:
        kf = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=random_state)
        # StratifiedKFold ha bisogno di y per bilanciare le classi
        split_generator = kf.split(X, y) 
    else:
        kf = KFold(n_splits=k, shuffle=shuffle, random_state=random_state)
        split_generator = kf.split(X)
        
    return split_generator