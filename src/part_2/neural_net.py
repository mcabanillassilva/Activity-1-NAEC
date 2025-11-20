import numpy as np

class NeuralNet:
    def __init__(self, layers, epochs=100, lr=0.01, momentum=0.9, activation='sigmoid', val_percent=0.2, scale=True):
        self.L = len(layers)    
        self.n = layers.copy()          
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.fact = activation
        self.val_percent = val_percent
        self.scale = scale

        # Initialize weights and thresholds
        self.w = [np.zeros((1,1))] 
        self.theta = [np.zeros((1,1))]
        for l in range(1, self.L):
            if activation == 'sigmoid':
                scale_factor = 0.01
            elif activation == 'tanh':
                scale_factor = np.sqrt(1.0 / self.n[l-1]) 
            elif activation == 'relu':
                scale_factor = np.sqrt(2.0 / self.n[l-1])
            elif activation == 'linear':
                scale_factor = 1e-5
            else:
                scale_factor = 0.01

            self.w.append(np.random.randn(self.n[l], self.n[l-1]) * scale_factor)
            self.theta.append(np.zeros((self.n[l], 1)))

        # Initialize activations, fields, errors, and weight deltas
        self.xi = [np.zeros((n,1)) for n in self.n]
        self.h = [np.zeros((n,1)) for n in self.n]
        self.delta = [np.zeros((n,1)) for n in self.n]
        self.d_w = [np.zeros((1,1))] + [np.zeros_like(self.w[l]) for l in range(1, self.L)]
        self.d_theta = [np.zeros((1,1))] + [np.zeros_like(self.theta[l]) for l in range(1, self.L)]
        self.d_w_prev = [np.zeros((1,1))] + [np.zeros_like(self.w[l]) for l in range(1, self.L)]
        self.d_theta_prev = [np.zeros((1,1))] + [np.zeros_like(self.theta[l]) for l in range(1, self.L)]
        
        self.train_errors = []
        self.val_errors = []
        

    def activate(self, x):
        if self.fact == 'sigmoid':
            return 1/(1+np.exp(-x))
        elif self.fact == 'tanh':
            return np.tanh(x)
        elif self.fact == 'relu':
            return np.maximum(0, x)
        elif self.fact == 'linear':
            return x
        else:
            raise ValueError("Unknown activation function")

    def activate_derivative(self, x):
        if self.fact == 'sigmoid':
            return x * (1 - x)
        elif self.fact == 'tanh':
            return 1 - x**2
        elif self.fact == 'relu':
            return np.where(x>0,1,0)
        elif self.fact == 'linear':
            return np.ones_like(x)
        else:
            raise ValueError("Unknown activation function")

    def forward(self, xi_input):
        """Forward pass for one sample"""
        # Scale input if needed
        if self.scale and self.x_mean is not None:
            xi_input = (xi_input - self.x_mean) / self.x_std

        self.xi[0] = xi_input.reshape(-1,1)  # input layer
        for l in range(1, self.L):
            self.h[l] = self.w[l] @ self.xi[l-1] - self.theta[l]
            self.xi[l] = self.activate(self.h[l])
        return self.xi[-1]

    def backward(self, yi):
        """Backpropagation for one sample"""
        yi = yi.reshape(-1,1)
        # Output layer delta
        self.delta[-1] = (yi - self.xi[-1]) * self.activate_derivative(self.xi[-1])
        # Hidden layers
        for l in reversed(range(1, self.L-1)):
            self.delta[l] = (self.w[l+1].T @ self.delta[l+1]) * self.activate_derivative(self.xi[l])
        # Weight and threshold updates
        for l in range(1, self.L):
            self.d_w[l] = self.lr * self.delta[l] @ self.xi[l-1].T + self.momentum * self.d_w_prev[l]
            self.d_theta[l] = self.lr * (-self.delta[l]) + self.momentum * self.d_theta_prev[l]

            self.w[l] += self.d_w[l]
            self.theta[l] += self.d_theta[l]

            self.d_w_prev[l] = self.d_w[l].copy()
            self.d_theta_prev[l] = self.d_theta[l].copy()

    def fit(self, X, y):
        """Train the network"""
        # Scale input features
        if self.scale:
            self.x_mean = X.mean(axis=0)
            self.x_std = X.std(axis=0)
            self.x_std[self.x_std==0] = 1.0  # prevent division by zero
            Xs = (X - self.x_mean)/self.x_std
        else:
            Xs = X.copy()
        ys = y.copy()

        # Split into validation if needed
        if self.val_percent > 0:
            val_size = int(len(Xs)*self.val_percent)
            X_val, X_train = Xs[:val_size], Xs[val_size:]
            y_val, y_train = ys[:val_size], ys[val_size:]
        else:
            X_train, y_train = Xs, ys
            X_val, y_val = None, None

        # Training loop
        for epoch in range(self.epochs):
            train_error = 0
            for xi_sample, yi_sample in zip(X_train, y_train):
                self.forward(xi_sample)
                self.backward(yi_sample)
                train_error += 0.5 * np.sum((yi_sample.reshape(-1,1)-self.xi[-1])**2)
            train_error /= len(X_train)
            self.train_errors.append(train_error)

            # Validation error
            if self.val_percent > 0:
                val_error = 0
                for xi_sample, yi_sample in zip(X_val, y_val):
                    yi_hat = self.forward(xi_sample)
                    val_error += 0.5 * np.sum((yi_sample.reshape(-1,1)-yi_hat)**2)
                val_error /= len(X_val)
                self.val_errors.append(val_error)

    def predict(self, X):
        if self.scale and self.x_mean is not None:
            Xs = (X - self.x_mean)/self.x_std
        else:
            Xs = X.copy()
        preds = []
        for xi_sample in Xs:
            yi_hat = self.forward(xi_sample)
            preds.append(yi_hat.flatten())
        return np.array(preds)
    
    def loss_epochs(self):
        np.set_printoptions(suppress=True)

        epochs = np.arange(1, self.epochs + 1).reshape(-1,1)

        train = np.zeros((self.epochs, 2))
        train[:,0] = np.array(self.train_errors).astype(float)
        train[:,1] = epochs[:,0].astype(float)

        if len(self.val_errors) == self.epochs:
            val = np.zeros((self.epochs, 2))
            val[:,0] = np.array(self.val_errors).astype(float)
            val[:,1] = epochs[:,0].astype(float)
        else:
            val = None

        return train, val

