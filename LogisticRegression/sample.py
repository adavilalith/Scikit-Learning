from LogisticRegression import LogisticRegression
import numpy as np
import pandas as pd

model = LogisticRegression()

data = pd.read_csv("./datasets/Iris.csv")
data["Species"] = data["Species"].apply(lambda x: 1 if x=="Iris-setosa" else 0)
data = data.drop(["Id"],axis=1)
x = data.drop(["Species"],axis=1) #input data
y = data["Species"] #ground truth (0 = edible or 1 = poisonous)
X = x.to_numpy().astype(np.float64)
Y = y.to_numpy().astype(np.float64)

x_train,x_test = X[:120],X[120:]
y_train,y_test = Y[:120],Y[120:]
model.fit(x_train,y_train)
pred = model.predict(x_test)
def accuracy(ground_truth,prediction):
    return sum(ground_truth==prediction)/len(ground_truth)
print("accuracy: ",accuracy(y_test,pred))