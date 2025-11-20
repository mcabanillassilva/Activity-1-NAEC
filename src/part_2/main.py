from train_bp import TrainBP
import numpy as np
import pickle

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

def run_training(trainer):
    nn, pid_test, y_test, preds_test, test_error = trainer.train(
        CSV_TRAIN,
        CSV_VAL,
        CSV_TEST,
        TARGET
    )

    for pid_val, real, pred in zip(pid_test, y_test, preds_test):
        print(f"PID {pid_val} - Real: {real:.2f}, Predicted: {pred:.2f}")

    print("Test quadratic error:", test_error)


def main():
    trainer = train_linear()
    run_training(trainer)


if __name__ == "__main__":
    main()

