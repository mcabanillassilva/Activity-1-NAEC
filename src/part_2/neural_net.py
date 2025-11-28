import numpy as np


class NeuralNet:
    def __init__(
        self,
        layers,
        epochs=100,
        lr=0.01,
        momentum=0.9,
        activation="sigmoid",
        val_percent=0.2,
        scale=True,
        l2_lambda=0.0,
        dropout_rate=0.0,
    ):
        self.L = len(layers)
        self.n = layers.copy()
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.fact = activation
        self.val_percent = val_percent
        self.scale = scale
        self.l2_lambda = l2_lambda  # L2 regularization parameter
        self.dropout_rate = dropout_rate  # Dropout rate (0 = no dropout)

        # Initialize weights and thresholds
        self.w = [np.zeros((1, 1))]
        self.theta = [np.zeros((1, 1))]
        for l in range(1, self.L):
            if activation == "sigmoid":
                scale_factor = 0.01
            elif activation == "tanh":
                scale_factor = np.sqrt(1.0 / self.n[l - 1])
            elif activation == "relu":
                scale_factor = np.sqrt(2.0 / self.n[l - 1])
            elif activation == "linear":
                scale_factor = 1e-5
            else:
                scale_factor = 0.01

            self.w.append(np.random.randn(self.n[l], self.n[l - 1]) * scale_factor)
            self.theta.append(np.zeros((self.n[l], 1)))

        # Initialize activations, fields, errors, and weight deltas
        self.xi = [np.zeros((n, 1)) for n in self.n]
        self.h = [np.zeros((n, 1)) for n in self.n]
        self.delta = [np.zeros((n, 1)) for n in self.n]
        self.d_w = [np.zeros((1, 1))] + [
            np.zeros_like(self.w[l]) for l in range(1, self.L)
        ]
        self.d_theta = [np.zeros((1, 1))] + [
            np.zeros_like(self.theta[l]) for l in range(1, self.L)
        ]
        self.d_w_prev = [np.zeros((1, 1))] + [
            np.zeros_like(self.w[l]) for l in range(1, self.L)
        ]
        self.d_theta_prev = [np.zeros((1, 1))] + [
            np.zeros_like(self.theta[l]) for l in range(1, self.L)
        ]

        # Dropout masks for each layer
        self.dropout_masks = [np.ones((n, 1)) for n in self.n]

        self.train_errors = []
        self.val_errors = []

    def activate(self, x):
        if self.fact == "sigmoid":
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif self.fact == "tanh":
            return np.tanh(x)
        elif self.fact == "relu":
            return np.maximum(0, x)
        elif self.fact == "linear":
            return x
        else:
            raise ValueError("Unknown activation function")

    def activate_derivative(self, x):
        if self.fact == "sigmoid":
            return x * (1 - x)
        elif self.fact == "tanh":
            return 1 - x**2
        elif self.fact == "relu":
            return np.where(x > 0, 1, 0)
        elif self.fact == "linear":
            return np.ones_like(x)
        else:
            raise ValueError("Unknown activation function")

    def apply_dropout(self, layer_idx, training=True):
        if training and self.dropout_rate > 0 and 0 < layer_idx < self.L - 1:
            mask = np.random.binomial(
                1, 1 - self.dropout_rate, size=self.xi[layer_idx].shape
            )
            self.dropout_masks[layer_idx] = mask / (
                1 - self.dropout_rate
            )  # inverted dropout
            self.xi[layer_idx] *= self.dropout_masks[layer_idx]

    def forward(self, xi_input, training=False):
        if self.scale and hasattr(self, "x_mean"):
            xi_input = (xi_input - self.x_mean) / self.x_std

        self.xi[0] = xi_input.reshape(-1, 1)
        for l in range(1, self.L):
            z = self.w[l] @ self.xi[l - 1] - self.theta[l]
            a = self.activate(z)

            # Dropout applied to activations
            if training and 0 < l < self.L - 1 and self.dropout_rate > 0:
                mask = np.random.binomial(1, 1 - self.dropout_rate, size=a.shape)
                mask = mask / (1 - self.dropout_rate)
                self.dropout_masks[l] = mask
                a *= mask
            else:
                self.dropout_masks[l] = np.ones_like(a)

            self.h[l] = z
            self.xi[l] = a

        return self.xi[-1]

    def backward(self, yi):
        yi = yi.reshape(-1, 1)
        self.delta[-1] = (yi - self.xi[-1]) * self.activate_derivative(self.xi[-1])

        for l in reversed(range(1, self.L - 1)):
            self.delta[l] = (
                self.w[l + 1].T @ self.delta[l + 1]
            ) * self.activate_derivative(self.xi[l])
            self.delta[l] *= self.dropout_masks[l]

        for l in range(1, self.L):
            # Correct L2 weight decay
            grad = self.delta[l] @ self.xi[l - 1].T
            grad += self.l2_lambda * self.w[l]  # weight decay
            self.d_w[l] = self.lr * grad + self.momentum * self.d_w_prev[l]
            self.d_theta[l] = (
                self.lr * (-self.delta[l]) + self.momentum * self.d_theta_prev[l]
            )

            self.w[l] += self.d_w[l]
            self.theta[l] += self.d_theta[l]

            self.d_w_prev[l] = self.d_w[l].copy()
            self.d_theta_prev[l] = self.d_theta[l].copy()

    def compute_l2_loss(self):
        l2_loss = 0
        for l in range(1, self.L):
            l2_loss += np.sum(self.w[l] ** 2)
        return 0.5 * self.l2_lambda * l2_loss

    def fit(self, X, y):
        if self.scale:
            self.x_mean = X.mean(axis=0)
            self.x_std = X.std(axis=0)
            self.x_std[self.x_std == 0] = 1.0
            Xs = (X - self.x_mean) / self.x_std
        else:
            Xs = X.copy()
        ys = y.copy()

        if self.val_percent > 0:
            val_size = int(len(Xs) * self.val_percent)
            X_val, X_train = Xs[:val_size], Xs[val_size:]
            y_val, y_train = ys[:val_size], ys[val_size:]
        else:
            X_train, y_train = Xs, ys
            X_val, y_val = None, None

        for epoch in range(self.epochs):
            train_error = 0
            for xi_sample, yi_sample in zip(X_train, y_train):
                self.forward(xi_sample, training=True)
                self.backward(yi_sample)
                train_error += 0.5 * np.sum(
                    (yi_sample.reshape(-1, 1) - self.xi[-1]) ** 2
                )

            # MSE real (no L2)
            mse_train = train_error / len(X_train)
            self.train_errors.append(mse_train)

            if self.val_percent > 0:
                val_error = 0
                for xi_sample, yi_sample in zip(X_val, y_val):
                    yi_hat = self.forward(xi_sample, training=False)
                    val_error += 0.5 * np.sum((yi_sample.reshape(-1, 1) - yi_hat) ** 2)
                mse_val = val_error / len(X_val)
                self.val_errors.append(mse_val)

            # Print progress
            if (epoch + 1) % 20 == 0:
                if self.val_percent > 0:
                    print(
                        f"Epoch {epoch+1}/{self.epochs} - Train Error: {mse_train:.6f}, Val Error: {mse_val:.6f}"
                    )
                else:
                    print(
                        f"Epoch {epoch+1}/{self.epochs} - Train Error: {mse_train:.6f}"
                    )

    def predict(self, X):
        if self.scale and hasattr(self, "x_mean"):
            Xs = (X - self.x_mean) / self.x_std
        else:
            Xs = X.copy()
        preds = [self.forward(xi_sample, training=False).flatten() for xi_sample in Xs]
        return np.array(preds)

    def loss_epochs(self):
        epochs = np.arange(1, self.epochs + 1).reshape(-1, 1)
        train = np.zeros((self.epochs, 2))
        train[:, 0] = np.array(self.train_errors)
        train[:, 1] = epochs[:, 0]
        if len(self.val_errors) == self.epochs:
            val = np.zeros((self.epochs, 2))
            val[:, 0] = np.array(self.val_errors)
            val[:, 1] = epochs[:, 0]
        else:
            val = None
        return train, val
