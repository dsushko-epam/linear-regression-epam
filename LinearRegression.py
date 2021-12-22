import numpy as np

class LinearRegression():
    """
    LSS-based linear regression model.
    """
    def __init__(self, method='analytical', 
                       learning_rate=0.01, 
                       max_epochs=100, 
                       early_stopping_threshold=0.0003, 
                       iters_to_stop=5,
                       regularization='none',
                       C=1.):
        """
        Initalizes new LinearResression model instance.
        -----------
        Parameters:
        -----------
            method: {'analytical', 'gd'}, default='analytical'
                Algorithm that is selected to calculate linear model
                weights and bias.
                'analytical' - explicit formula (A^T*A)^-1*A^T*y
                'gd' - gradient descent
            learning_rate: float, default=0.01
                If method='gd', then defines learning rate
                for gradient descent, else ignored
            max_epochs: int, default=100
                For method='gd' defines upper boundary for
                how many iterations will be done.
            early_stopping_threshold: float, default=0.0003
                For method='gd' defines what loss function value
                difference between epoch and epoch+1 is considered as
                idle iteration and convergence (criteria to stop).
            iters_to_stop: int, default=5
                For method='gd' defines how many idle iterations in a row
                is required to stop further calculations.
            regularization: {'none', 'l2', 'l1'}, default='none'
                Type of regularization used in linear model.
                'l1' works only for method='gd'
            C: float, default=1.0
                Regularization parameter. Ignored, if
                regularization=none
        """
        self.method = method
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.early_stopping_threshold = early_stopping_threshold
        self.iters_to_stop = iters_to_stop
        self.c = C
        pass

    def fit(self, X, y):
        """
        Fits X to y: calculates weights to minimize
        MSE related to y.
        Returns self
        """

        if self.regularization == 'l1' and self.method != 'gd':
            print('Warning! Regularization parameter is set to \'l1\' whilst method is not \'gd\'! Using method=\'gd\'...')
            self.method = 'gd'
        
        if self.method == 'gd':
            self.coef_ = np.zeros(X.shape[1])
            self.bias = 0
            a_ddx = np.zeros(X.shape[1])
            b_ddx = 0

            self.prev_error = 0
            self.useless_iterations = 0

            for ep in range(self.max_epochs):
                y_pred = X @ self.coef_ + self.bias
                error = y - y_pred.ravel()
                # early stopping
                if np.abs(error.mean()-self.prev_error) < self.early_stopping_threshold:
                    if self.useless_iterations >= self.iters_to_stop:
                        return self
                    self.useless_iterations += 1
                else:
                    self.useless_iterations = 0
                self.prev_error_mean = error.mean()
                
                for i in range(X.shape[1]):
                    if self.regularization == 'l2':
                        a_ddx[i] = -2 * (X[:,i] * error).mean() + 2*self.c*self.coef_[i]/len(X[:,i])
                    if self.regularization == 'l1':
                        a_ddx[i] = -2 * (X[:,i] * error).mean() + np.sign(self.coef_[i])*self.c/len(X[:,i])
                    if self.regularization == 'none':
                        a_ddx[i] = -2 * (X[:,i] * error).mean()
                b_ddx = -2 * error.mean()
                     
                for i in range(X.shape[1]):
                    self.coef_[i] -= self.learning_rate * a_ddx[i]
                self.bias -= self.learning_rate * b_ddx
            return self
        if self.method == 'analytical':
            ones = np.ones(shape=(X.shape[0], 1))
            A = np.concatenate([ones, X], axis=1)
            if self.regularization == 'none':
                weights = np.linalg.inv(A.T @ A) @ A.T @ y
                self.coef_ = weights[1:]
                self.bias = weights[0]
            if self.regularization == 'l2':
                weights = np.linalg.inv(A.T @ A + self.c*np.eye(A.shape[1])) @ A.T @ y
                self.coef_ = weights[1:]
                self.bias = weights[0]            
        return self

    def predict(self, X):
        """
        Predicts and returns outputs for X using formula:
            ---------------
            y_pred = Xw + b
            ---------------
            X - input data, 2-dimensional array 
            (samples x features).
            w, b - vector of weights and bias scalar
            calculated at fit() step.
        """
        return X @ self.coef_ + self.bias