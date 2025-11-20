import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

try:
    from .train_bp import TrainBP
except ImportError:
    from train_bp import TrainBP

TARGET = "SalePrice"

CSV_TRAIN = "output/train_processed.csv"
CSV_VAL = "output/val_processed.csv"
CSV_TEST = "output/test_processed.csv"
SCALER_Y_PATH = "src/part_1/scaler_y.pkl"

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
        log_target=False
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
        log_target=True
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
        log_target=True
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
        log_target=False
    )
    return trainer

def run_training(activation: str, verbose: bool = False):
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
    nn, pid_test, y_test, preds_test, test_error = trainer.train(
        CSV_TRAIN,
        CSV_VAL,
        CSV_TEST,
        TARGET,
        verbose=verbose
    )
    # Just for checking results
    if verbose:
        for pid_val, real, pred in zip(pid_test, y_test, preds_test):
            print(f"PID {pid_val} - Real: {real:.2f}, Predicted: {pred:.2f}")

    return nn, pid_test, y_test, preds_test, trainer

def main():
    run_training("linear")


if __name__ == "__main__":
    main()

