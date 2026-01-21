# training/model_factory.py
from nn.model import Model
from nn.layers import Dense
from nn.dropout import Dropout
from nn.activations import Tanh, Sigmoid, ReLU, Identity, Softmax
from nn.losses import BinaryCrossEntropy, CrossEntropy, MSE
from nn.metrics import Accuracy, MSE as MSEMetric
from nn.optim import SGD, SGDMomentum, Adam
from nn.callbacks import EarlyStopping


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

    modules = []
    prev = int(in_dim)
    for h in hidden_list:
        modules.append(Dense(prev, int(h), seed=seed))
        modules.append(_make_activation(activation_name))
        if dropout_p > 0.0:
            modules.append(Dropout(p=dropout_p, seed=seed))
        prev = int(h)

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
        metrics = [MSEMetric()]
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