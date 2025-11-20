import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.part_2.main import run_training

def main():
    activation = "linear" # ["linear", "sigmoid", "tanh", "relu"]
    nn, pid_test, y_test, preds_test, trainer = run_training(activation, False)
    
    pred_BP = preds_test
    
    print("Predictions from BP Neural Network: ", pred_BP)
    
    
if __name__ == "__main__":
    main()