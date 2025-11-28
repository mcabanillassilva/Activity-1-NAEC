import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

try:
    from .train_housing_bp import TrainBP
except ImportError:
    from src.part_2.train_housing_bp import TrainBP

TARGET = "SalePrice"


def train_linear():
    layers = [277, 1]
    trainer = TrainBP(
        layers=layers,
        epochs=200,
        lr=1e-6,
        momentum=0,
        activation="linear",
        val_percent=0.2,
        scale=True,
        log_target=False,
    )
    return trainer


def train_sigmoid():
    layers = [277, 20, 10, 1]
    trainer = TrainBP(
        layers=layers,
        epochs=70,
        lr=0.001,
        momentum=0.9,
        activation="sigmoid",
        val_percent=0.2,
        scale=True,
        log_target=True,
    )
    return trainer


def train_tanh():
    layers = [277, 20, 10, 1]
    trainer = TrainBP(
        layers=layers,
        epochs=200,
        lr=0.05,
        momentum=0.9,
        activation="tanh",
        val_percent=0.2,
        scale=True,
        log_target=True,
    )
    return trainer


def train_relu():
    layers = [277, 64, 32, 1]
    trainer = TrainBP(
        layers=layers,
        epochs=200,
        lr=1e-4,
        momentum=0.9,
        activation="relu",
        val_percent=0.2,
        scale=True,
        log_target=False,
    )
    return trainer


def run_training(
    activation: str,
    verbose: bool = False,
    CSV_TRAIN: str = None,
    CSV_VAL: str = None,
    CSV_TEST: str = None,
    # --- Optional hyperparameters for override ---
    layers: list = None,
    epochs: int = None,
    lr: float = None,
    momentum: float = None,
    val_percent: float = None,
    scale: bool = None,
    log_target: bool = None,
):
    # 1. Load default trainer depending on activation
    if activation == "linear":
        trainer = train_linear()
    elif activation == "sigmoid":
        trainer = train_sigmoid()
    elif activation == "tanh":
        trainer = train_tanh()
    elif activation == "relu":
        trainer = train_relu()
    else:
        raise ValueError(f"Unknown activation function: {activation}")

    if layers is not None:
        trainer.layers = layers

    if epochs is not None:
        trainer.epochs = epochs

    if lr is not None:
        trainer.lr = lr

    if momentum is not None:
        trainer.momentum = momentum

    if val_percent is not None:
        trainer.val_percent = val_percent

    if scale is not None:
        trainer.scale = scale

    if log_target is not None:
        trainer.log_target = log_target

    # 2. Train the model
    nn, pid_test, y_test, preds_test, test_error = trainer.train(
        CSV_TRAIN, CSV_VAL, CSV_TEST, TARGET, verbose=verbose
    )

    # 4. Optional verbose output
    if verbose:
        for pid_val, real, pred in zip(pid_test, y_test, preds_test):
            print(f"PID {pid_val} - Real: {real:.2f}, Predicted: {pred:.2f}")

    return nn, pid_test, y_test, preds_test, trainer


def loss_epochs(trainer: TrainBP):
    """Return training and validation loss per epoch as a DataFrame"""
    return trainer.loss_epochs()


def main():
    CSV_TRAIN = "output/train_processed.csv"
    CSV_VAL = "output/val_processed.csv"
    CSV_TEST = "output/test_processed.csv"
    run_training(
        "linear", verbose=True, CSV_TRAIN=CSV_TRAIN, CSV_VAL=CSV_VAL, CSV_TEST=CSV_TEST
    )


if __name__ == "__main__":
    main()
