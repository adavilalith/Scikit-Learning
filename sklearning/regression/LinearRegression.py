import numpy as np
import pandas as pd
class LinearRegression:
    def __init__(self,lr=0.001,iters=100):
        self.lr=lr
        self.iters = iters

    def fit(self,X,Y,iters=100):
        self.iters=iters
        self.w = np.zeros(X.shape[1])
        b = 0
        for _ in range(iters):
            y_pred = np.dot(X,self.w) + b
            db = 2/X.shape[0] * np.sum(y_pred-Y)
            dw = 2/X.shape[0] * np.dot((y_pred-Y),X)
            b = b - self.lr * db
            self.w = self.w - self.lr * dw
    
    def predict(self,X):
        return np.dot(X,self.w) + self.b

