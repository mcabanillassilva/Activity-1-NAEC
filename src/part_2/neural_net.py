import numpy as np

class NeuralNet:
    def __init__(self, layers):
        self.L = len(layers)
        self.n = layers.copy()

        self.xi = [np.zeros(l) for l in layers]

        self.w = []
        self.w.append(np.zeros((1, 1)))
        for lay in range(1, self.L):
            self.w.append(np.zeros((layers[lay], layers[lay - 1])))

layers = [4, 9, 5, 1]
nn = NeuralNet(layers)

print("L =", nn.L)
print("n =", nn.n)
print("xi =", nn.xi)
print("w[1] =", nn.w[1])
