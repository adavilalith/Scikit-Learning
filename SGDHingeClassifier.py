from sklearning.classification import SGDHingeClassifier
import numpy as np
import pandas as pd

model = SGDHingeClassifier()

data = pd.read_csv("./datasets/mushrooms.csv")
data = data.sample(frac=1).reset_index(drop=True)
x = data.drop(["class"],axis=1) #input data
y = data["class"] 
from sklearning.utils.LabelEncoder import LabelEncoder
transformed_x = pd.DataFrame()
for c in x:
    le =  LabelEncoder(1)
    transformed_x[c] =  le.fit_transform(x[c])

from sklearning.utils.StandardScaler import StandardScaler
sc = StandardScaler()
scaled_x = sc.transform(transformed_x)

X = scaled_x.to_numpy().astype(np.int64)
Y = np.array([-1 if i=="e" else 1 for i in y]).astype(np.int64)
x_train,x_test = X[:120],X[120:]
y_train,y_test = Y[:120],Y[120:]

model.fit(x_train,y_train)
pred = model.predict(x_test)
def accuracy(ground_truth,prediction):
    return sum(ground_truth==prediction)/len(ground_truth)
print("accuracy: ",accuracy(y_test,pred))

data = pd.read_csv("./datasets/Iris.csv")
data = data.sample(frac=1).reset_index(drop=True)
data["Species"] = data["Species"].apply(lambda x: 1 if x=="Iris-setosa" else -1)
data = data.drop(["Id"],axis=1)
x = data.drop(["Species"],axis=1) #input data
y = data["Species"] #ground truth (0 = edible or 1 = poisonous)
X = x.to_numpy().astype(np.float64)
Y = y.to_numpy().astype(np.float64)

x_train,x_test = X[:120],X[120:]
y_train,y_test = Y[:120],Y[120:]

model = SGDHingeClassifier()

model.fit(x_train,y_train)
pred = model.predict(x_test)
def accuracy(ground_truth,prediction):
    return sum(ground_truth==prediction)/len(ground_truth)
print("accuracy: ",accuracy(y_test,pred))