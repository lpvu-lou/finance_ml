import numpy as np

class LinearRegression:
    def __init__(self, use_intercept: bool = True):
        self.use_intercept = use_intercept
        self.coef = None
        self.intercept = None
    
    def fit(self, X, y):
        if self.use_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        beta = np.linalg.pinv(X.T @ X) @ X.T @ y

        if self.use_intercept:
            self.intercept = beta[0]
            self.coef = beta[1:]
        else:
            self.coef = beta

        return self

    def predict(self, X): 
        X = np.array(X)
        if self.use_intercept:
            return X @ self.coef + self.intercept
        return X @ self.coef
    

