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
        l2_lambda=0.0,
        dropout_rate=0.0,
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
        l2_lambda=0.0,
        dropout_rate=0.0,
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
        l2_lambda=0.0,
        dropout_rate=0.0,
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
        l2_lambda=0.0,
        dropout_rate=0.0,
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
    l2_lambda: float = None,  # NEW: L2 regularization
    dropout_rate: float = None,  # NEW: Dropout regularization
):
    """
    Run training with optional regularization parameters.

    Parameters:
    -----------
    activation : str
        Activation function ('linear', 'sigmoid', 'tanh', 'relu')
    verbose : bool
        Print detailed output
    CSV_TRAIN, CSV_VAL, CSV_TEST : str
        Paths to data files
    layers : list
        Network architecture (e.g., [277, 20, 10, 1])
    epochs : int
        Number of training epochs
    lr : float
        Learning rate
    momentum : float
        Momentum coefficient
    val_percent : float
        Validation split percentage
    scale : bool
        Whether to scale features
    log_target : bool
        Whether to apply log transform to target
    l2_lambda : float
        L2 regularization parameter (0.0 = no regularization)
        Typical values: 0.0001, 0.001, 0.01
    dropout_rate : float
        Dropout rate (0.0 = no dropout, 0.5 = 50% dropout)
        Typical values: 0.0, 0.1, 0.2, 0.3, 0.5

    Returns:
    --------
    nn : NeuralNet
        Trained neural network
    pid_test : array
        Test set property IDs
    y_test : array
        True test labels
    preds_test : array
        Predicted test labels
    trainer : TrainBP
        Trainer object with training history
    """
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

    # 2. Override hyperparameters if provided
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

    if l2_lambda is not None:
        trainer.l2_lambda = l2_lambda
    if dropout_rate is not None:
        trainer.dropout_rate = dropout_rate

    if verbose:
        print("\n" + "=" * 70)
        print("TRAINING CONFIGURATION")
        print("=" * 70)
        print(f"Activation: {activation}")
        print(f"Architecture: {trainer.layers}")
        print(f"Epochs: {trainer.epochs}")
        print(f"Learning Rate: {trainer.lr}")
        print(f"Momentum: {trainer.momentum}")
        print(f"L2 Lambda: {trainer.l2_lambda}")
        print(f"Dropout Rate: {trainer.dropout_rate}")
        print(f"Scale: {trainer.scale}")
        print(f"Log Target: {trainer.log_target}")
        print("=" * 70 + "\n")

    nn, pid_test, y_test, preds_test, test_error = trainer.train(
        CSV_TRAIN, CSV_VAL, CSV_TEST, TARGET, verbose=verbose
    )

    if verbose:
        print("\n" + "=" * 70)
        print("TEST SET PREDICTIONS (Sample)")
        print("=" * 70)
        for i, (pid_val, real, pred) in enumerate(
            zip(pid_test[:10], y_test[:10], preds_test[:10])
        ):
            print(
                f"PID {pid_val} - Real: ${real:,.2f}, Predicted: ${pred:,.2f}, Error: ${abs(real-pred):,.2f}"
            )
        if len(pid_test) > 10:
            print(f"... ({len(pid_test)-10} more predictions)")
        print("=" * 70 + "\n")

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
