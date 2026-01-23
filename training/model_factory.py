# training/model_factory.py
from nn.model import Model
from nn.layers import Dense
from nn.dropout import Dropout
from nn.activations import Tanh, Sigmoid, ReLU, Identity, Softmax
from nn.losses import BinaryCrossEntropy, CrossEntropy, MSE
from nn.metrics import Accuracy, MSE as MSEMetric, MEE
from nn.optim import SGD, SGDMomentum, Adam
from nn.callbacks import EarlyStopping
from nn.initializers import xavier_uniform, he_uniform  # <--- NUOVO IMPORT


def _make_activation(name):
    n = str(name).lower()
    if n == "tanh":
        return Tanh()
    if n == "sigmoid":
        return Sigmoid()
    if n == "relu":
        return ReLU()
    if n == "identity":
        return Identity()
    if n == "softmax":
        return Softmax()
    raise ValueError(f"Unknown activation: {name!r}")


def _select_initializer(activation_name):
    """
    Select appropriate weight initializer based on activation function.
    
    Args:
        activation_name: Name of activation function (str)
    
    Returns:
        Initializer function (xavier_uniform or he_uniform)
    
    Rules:
        - ReLU → He initialization (accounts for rectification)
        - Tanh, Sigmoid, Identity → Xavier initialization (symmetric functions)
    """
    n = str(activation_name).lower()
    if n == "relu":
        return he_uniform
    else:
        # Xavier for tanh, sigmoid, identity, and other symmetric activations
        return xavier_uniform


def build_model_from_cfg(run_cfg, seed, in_dim, out_dim, task="binary"):
    """Generic sequential model builder driven by run_cfg."""
    model_cfg = run_cfg.get("model", {})
    optim_cfg = run_cfg.get("optim", {})
    reg_cfg = run_cfg.get("regularizer", {})
    cb_cfg = run_cfg.get("callbacks", {})

    hidden_units = model_cfg.get("hidden_units", 16)
    if isinstance(hidden_units, int):
        hidden_list = [hidden_units]
    else:
        hidden_list = [int(x) for x in hidden_units]

    activation_name = str(model_cfg.get("activation", "tanh"))
    dropout_p = float(model_cfg.get("dropout", 0.0))

    # NUOVO: Seleziona initializer basato su activation
    initializer = _select_initializer(activation_name)

    modules = []
    prev = int(in_dim)
    for h in hidden_list:
        # Passa initializer appropriato al Dense layer
        modules.append(Dense(prev, int(h), initializer=initializer, seed=seed))
        modules.append(_make_activation(activation_name))
        if dropout_p > 0.0:
            modules.append(Dropout(p=dropout_p, seed=seed))
        prev = int(h)

    task = str(task).lower()
    if task == "binary":
        # Output layer usa Xavier (sigmoid è simmetrica)
        modules.append(Dense(prev, int(out_dim), initializer=xavier_uniform, seed=seed))
        modules.append(Sigmoid())
        loss = BinaryCrossEntropy()
        metrics = [Accuracy(), MSE()]
    elif task == "multiclass":
        # Output layer usa Xavier (softmax è simmetrica)
        modules.append(Dense(prev, int(out_dim), initializer=xavier_uniform, seed=seed))
        modules.append(Softmax())
        loss = CrossEntropy()
        metrics = [Accuracy()]
    elif task == "regression":
        # Output layer usa Xavier (identity è lineare)
        modules.append(Dense(prev, int(out_dim), initializer=xavier_uniform, seed=seed))
        modules.append(Identity())
        loss = MSE()
        metrics = [MEE()]
    else:
        raise ValueError(f"Unknown task: {task!r}")

    lr = float(optim_cfg.get("lr", 0.1))
    momentum = float(optim_cfg.get("momentum", 0.0))
    opt_name = str(optim_cfg.get("name", "sgd")).lower()

    if opt_name == "adam":
        optimizer = Adam(lr=lr)
    else:
        optimizer = SGDMomentum(lr=lr, momentum=momentum) if momentum > 0.0 else SGD(lr=lr)

    regularizer = None
    lam = float(reg_cfg.get("l2", 0.0))
    if lam > 0.0:
        from nn.regularizers import L2
        regularizer = L2(lam=lam)

    callbacks = []
    if bool(cb_cfg.get("early_stopping", False)):
        callbacks.append(EarlyStopping(
            monitor=str(cb_cfg.get("monitor", "val_loss")),
            patience=int(cb_cfg.get("patience", 10)),
            min_delta=float(cb_cfg.get("min_delta", 0.0)),
            mode=str(cb_cfg.get("mode", "auto")),
            restore_best_weights=bool(cb_cfg.get("restore_best_weights", True)),
            verbose=int(cb_cfg.get("verbose", 0)),
        ))

    return Model(
        modules=modules,
        loss=loss,
        optimizer=optimizer,
        regularizer=regularizer,
        metrics=metrics,
        callbacks=callbacks,
    )