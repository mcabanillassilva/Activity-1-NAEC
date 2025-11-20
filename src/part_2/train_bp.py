import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from neural_net import NeuralNet

class TrainBP:
    """Class to handle training of a neural network using backpropagation and handle data loading"""
    def __init__(self, layers, epochs=200, lr=0.05, momentum=0.9,
                 activation='sigmoid', val_percent=0.2, scale=True, log_target=False):
        self.layers = layers
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.activation = activation
        self.val_percent = val_percent
        self.scale = scale          # scale inputs X
        self.log_target = log_target  # log-transform target
        self.nn = None
        self.scaler_X = None
        self.scaler_y = None

    def load_data(self, train_path, val_path, test_path, target):
        df_train = pd.read_csv(train_path)
        df_val = pd.read_csv(val_path)
        df_test = pd.read_csv(test_path)

        # Apply log transform if requested
        if self.log_target:
            df_train[target] = np.log1p(df_train[target].values)
            df_val[target]   = np.log1p(df_val[target].values)
            df_test[target]  = np.log1p(df_test[target].values)

        # Unique IDs
        pid_train = df_train['PID'].values
        pid_val = df_val['PID'].values
        pid_test = df_test['PID'].values

        # Separate features and target
        X_train = df_train.drop(columns=[target, 'PID']).values
        y_train = df_train[target].values
        X_val = df_val.drop(columns=[target, 'PID']).values
        y_val = df_val[target].values
        X_test = df_test.drop(columns=[target, 'PID']).values
        y_test = df_test[target].values
        
        return X_train, y_train, pid_train, X_val, y_val, pid_val, X_test, y_test, pid_test

    def scale_features(self, X_train, X_val, X_test):
        """Scale input features independently of the target"""
        if self.scale:
            self.scaler_X = StandardScaler()
            X_train_scaled = self.scaler_X.fit_transform(X_train)
            X_val_scaled   = self.scaler_X.transform(X_val)
            X_test_scaled  = self.scaler_X.transform(X_test)
            return X_train_scaled, X_val_scaled, X_test_scaled
        else:
            return X_train, X_val, X_test

    def scale_target(self, y_train, y_val):
        if self.activation in ['sigmoid', 'tanh']:
            self.scaler_y = MinMaxScaler()
            y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1,1)).flatten()
            y_val_scaled   = self.scaler_y.transform(y_val.reshape(-1,1)).flatten()
        elif self.activation in ['linear', 'relu']:
            self.scaler_y = StandardScaler()
            y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1,1)).flatten()
            y_val_scaled   = self.scaler_y.transform(y_val.reshape(-1,1)).flatten()
        else:
            self.scaler_y = None
            y_train_scaled, y_val_scaled = y_train, y_val
        return y_train_scaled, y_val_scaled


    def inverse_target(self, y_pred, y_true=None):
        if self.scaler_y:
            y_pred = self.scaler_y.inverse_transform(y_pred.reshape(-1,1)).flatten()
            if self.scale and y_true is not None:
                y_true = self.scaler_y.inverse_transform(y_true.reshape(-1,1)).flatten()
        if self.log_target:
            y_pred = np.expm1(y_pred)
            if y_true is not None:
                y_true = np.expm1(y_true)
        return y_pred, y_true


    def train(self, train_path, val_path, test_path, target):
        # Load data
        X_train, y_train, pid_train, X_val, y_val, pid_val, X_test, y_test, pid_test = \
            self.load_data(train_path, val_path, test_path, target)

        # Scale features
        X_train, X_val, X_test = self.scale_features(X_train, X_val, X_test)

        # Scale target if needed
        y_train_scaled, y_val_scaled = self.scale_target(y_train, y_val)
        
        # Scale y_test using the same scaler
        if self.scaler_y is not None:
            y_test_scaled = self.scaler_y.transform(y_test.reshape(-1,1)).flatten()
        else:
            y_test_scaled = y_test

        # Merge training and validation
        X_train_full = np.concatenate([X_train, X_val], axis=0)
        y_train_full = np.concatenate([y_train_scaled, y_val_scaled], axis=0)

        # Initialize and fit network
        self.nn = NeuralNet(
            self.layers,
            epochs=self.epochs,
            lr=self.lr,
            momentum=self.momentum,
            activation=self.activation,
            val_percent=self.val_percent,
            scale=False  # already scaled manually
        )
        self.nn.fit(X_train_full, y_train_full)

        # Predict
        preds_test = self.nn.predict(X_test)

        # Quadratic error
        test_error = 0.5 * np.mean((y_test_scaled - preds_test)**2)
        
        # Inverse transformations
        preds_test_original, y_test_original = self.inverse_target(preds_test, y_test_scaled)
   
        return self.nn, pid_test, y_test_original.flatten(), preds_test_original.flatten(), test_error
