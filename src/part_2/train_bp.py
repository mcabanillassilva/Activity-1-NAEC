import pandas as pd
from NeuralNet import NeuralNet

train = pd.read_csv("output/train_processed.csv")
val = pd.read_csv("output/val_processed.csv")

X_train = train.drop(columns=['SalePrice']).values
y_train = train['SalePrice'].values
X_val = val.drop(columns=['SalePrice']).values
y_val = val['SalePrice'].values

layers = [X_train.shape[1], 20, 10, 1]
nn = NeuralNet(layers)

print("Neural network initialized:")
print("Layers:", nn.n)
print("Weights shapes:", [w.shape for w in nn.w])
