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

        self.theta = [np.zeros(l) for l in layers]

        self.delta = [np.zeros(l) for l in layers]

        self.d_w = [np.zeros((1, 1))]
        self.d_theta = [np.zeros(l) for l in layers]

        self.d_w_prev = [np.zeros((1, 1))]
        self.d_theta_prev = [np.zeros(l) for l in layers]

layers = [4, 9, 5, 1]
nn = NeuralNet(layers)

print("L =", nn.L)
print("n =", nn.n)
print("theta[1] =", nn.theta[1])
print("delta[1] =", nn.delta[1])
print("d_w[1] =", nn.d_w)
print("d_theta_prev[1] =", nn.d_theta_prev[1])

