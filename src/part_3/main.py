import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.part_2.main import run_training
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from sklearn.neural_network import MLPRegressor
import pandas as pd
from sklearn.linear_model import LinearRegression

def mape(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape_val = mape(y_true, y_pred)
    return {
        "MSE": float(mse),
        "MAE": float(mae),
        "MAPE": float(mape_val)
    }


def train_bp_f(X_train, y_train, X_test, activation="linear"):
    """
    BP-F model using s MLPRegressor
    """
    mlp = MLPRegressor(
        hidden_layer_sizes=(20, 10),
        activation=activation,
        solver="adam",
        max_iter=200,
        random_state=42
    )
    mlp.fit(X_train, y_train)
    preds = mlp.predict(X_test)
    return preds, mlp


def train_mlr_f(X_train, y_train, X_test):
    """
    MLR-F model using scikit-learn's LinearRegression
    """
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    preds = lr.predict(X_test)
    return preds, lr

def main():
    # Activation functions to compare
    activation_bp = "linear"
    activation_bp_f = "identity"  # equivalent to 'linear' in sklearn
    
    # BP Custom Neural Network
    nn, pid_test, y_test, preds_bp, trainer = run_training(activation_bp, verbose=False)
    

    X_train = pd.read_csv("output/train_processed.csv").drop(columns=["SalePrice", "PID"])
    y_train = pd.read_csv("output/train_processed.csv")["SalePrice"]
    X_test = pd.read_csv("output/test_processed.csv").drop(columns=["SalePrice", "PID"])
    
    # BP-F 
    preds_bp_f, bp_f_model = train_bp_f(X_train, y_train, X_test, activation_bp_f)
    
    # Evaluate
    print("BP Custom:", evaluate_model(y_test, preds_bp))
    print("BP-F:", evaluate_model(y_test, preds_bp_f))

    
if __name__ == "__main__":
    main()