# training/model_factory.py
from nn.model import Model
from nn.layers import Dense
from nn.dropout import Dropout
from nn.activations import Tanh, Sigmoid, ReLU, Identity, Softmax
from nn.losses import BinaryCrossEntropy, CrossEntropy, MSE
from nn.metrics import Accuracy, MSE as MSEMetric, MEE
from nn.optim import SGD, SGDMomentum, Adam
from nn.regularizers import L2
from nn.callbacks import EarlyStopping


# Activation registry for cleaner lookup
_ACTIVATIONS = {
    "tanh": Tanh,
    "sigmoid": Sigmoid,
    "relu": ReLU,
    "identity": Identity,
    "softmax": Softmax,
}


def _make_activation(name):
    """Create activation instance from name (backward compatible)."""
    name = str(name).lower()
    
    if name not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation: {name!r}")
    
    return _ACTIVATIONS[name]()


def build_model_from_cfg(run_cfg, seed, in_dim, out_dim, task="binary"):
    """
    Generic sequential model builder driven by run_cfg.
    
    IMPORTANT: This function signature and config structure MUST NOT change
    to maintain compatibility with gridsearch and model_selection.
    """
    model_cfg = run_cfg.get("model", {})
    optim_cfg = run_cfg.get("optim", {})
    reg_cfg = run_cfg.get("regularizer", {})
    cb_cfg = run_cfg.get("callbacks", {})

    # Parse hidden layers config
    hidden_units = model_cfg.get("hidden_units", 16)
    if isinstance(hidden_units, int):
        hidden_list = [hidden_units]
    else:
        hidden_list = [int(x) for x in hidden_units]

    activation_name = model_cfg.get("activation", "tanh")
    dropout_p = model_cfg.get("dropout", 0.0)

    # Build hidden layers
    modules = []
    prev = int(in_dim)
    
    for h in hidden_list:
        modules.append(Dense(prev, int(h), seed=seed))
        modules.append(_make_activation(activation_name))
        if dropout_p > 0.0:
            modules.append(Dropout(p=dropout_p, seed=seed))
        prev = int(h)

    # Build output layer based on task
    task = str(task).lower()
    
    if task == "binary":
        modules.append(Dense(prev, int(out_dim), seed=seed))
        modules.append(Sigmoid())
        loss = BinaryCrossEntropy()
        metrics = [Accuracy()]
    
    elif task == "multiclass":
        modules.append(Dense(prev, int(out_dim), seed=seed))
        modules.append(Softmax())
        loss = CrossEntropy()
        metrics = [Accuracy()]
    
    elif task == "regression":
        modules.append(Dense(prev, int(out_dim), seed=seed))
        modules.append(Identity())
        loss = MSE()
        metrics = [MEE()]
    
    else:
        raise ValueError(f"Unknown task: {task!r}")

    # Build optimizer
    lr = optim_cfg.get("lr", 0.1)
    momentum = optim_cfg.get("momentum", 0.0)
    opt_name = optim_cfg.get("name", "sgd")
    
    if str(opt_name).lower() == "adam":
        optimizer = Adam(lr=lr)
    elif momentum > 0.0:
        optimizer = SGDMomentum(lr=lr, momentum=momentum)
    else:
        optimizer = SGD(lr=lr)

    # Build regularizer (optional)
    regularizer = None
    lam = reg_cfg.get("l2", 0.0)
    if lam > 0.0:
        regularizer = L2(lam=lam)

    # Build callbacks (optional)
    callbacks = []
    if cb_cfg.get("early_stopping", False):
        callbacks.append(EarlyStopping(
            monitor=cb_cfg.get("monitor", "val_loss"),
            patience=cb_cfg.get("patience", 10),
            min_delta=cb_cfg.get("min_delta", 0.0),
            mode=cb_cfg.get("mode", "auto"),
            restore_best_weights=cb_cfg.get("restore_best_weights", True),
            verbose=cb_cfg.get("verbose", 0),
        ))

    return Model(
        modules=modules,
        loss=loss,
        optimizer=optimizer,
        regularizer=regularizer,
        metrics=metrics,
        callbacks=callbacks,
    )
