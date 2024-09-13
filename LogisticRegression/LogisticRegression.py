import numpy as np
import pandas as pd

class LogisticRegression:
    def __init__(self,lr=0.001,iters=100):
        self.lr = lr
        self.iters = iters

    def sigmoid(self,z):
        return 1 / (1 + np.exp(-z))

    def calc(self,w,b,X):
        z=0
        for i in range(X.shape[1]):
            z += w[i] * X[:,i]
        z = np.array(b+z)
        return self.sigmoid(z)
    
    def gradient(self,w,b,X,Y):
        y_pred = self.calc(w,b,X)
        new_b = -1 * sum(Y*(1-y_pred) - (1-Y)*y_pred)
        new_w = np.zeros(w.shape[0])
        for i in range(w.shape[0]):
            new_w[i] = -1 * sum(Y*(1-y_pred)*X[:,i] - (1-Y)*y_pred*X[:,i])
        return new_w,new_b
    
    def fit(self,X,Y):
        self.no_of_records = X.shape[0]
        self.no_of_features = X.shape[1]
        self.w = np.zeros(self.no_of_features)
        self.b = 0
        for i in range(self.iters):
            new_w,new_b = self.gradient(self.w,self.b,X,Y)
            self.b = self.b - self.lr * new_b
            self.w = self.w - self.lr * new_w
    def predict(self,X):
        temp =  np.dot(X,self.w)+self.b
        return [1 if i>=0.5 else 0 for i in temp]
    

