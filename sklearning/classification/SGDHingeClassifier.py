import numpy as np
class SGDHingeClassifier:
    def __init__(self,lr=0.001,iters=100,lambda_param = 0.01):
        self.lr = lr
        self.lambda_param = lambda_param
        self.iters = iters
    
    def fit(self,X,Y):
        self.w = np.zeros(X.shape[1])
        self.b = 0 
        for _ in range(self.iters):
            #sgd
            for idx,x_i in enumerate(X):
                #hinge loss
                if Y[idx] * (np.dot(x_i,self.w) - self.b) >=1:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * ( 2 * self.lambda_param * self.w - np.dot(x_i,Y[idx]))
                    self.b -= self.lr * Y[idx]
    
    def predict(self,X):
        temp = np.dot(X,self.w)-self.b
        return np.sign(temp)
    

