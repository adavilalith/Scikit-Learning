import numpy as np
import pandas as pd

data = pd.read_csv("./datasets/CAR DETAILS FROM CAR DEKHO.xls")

from sklearning.utils.LabelEncoder import LabelEncoder
le = LabelEncoder()
data["label_encoded_transmission"] = le.fit_transform(data["transmission"]).astype(np.int32)
data["label_encoded_fuel"] = le.fit_transform(data["fuel"]).astype(np.int32)
data["label_encoded_seller_type"] = le.fit_transform(data["seller_type"]).astype(np.int32)
data["label_encoded_owner"] = le.fit_transform(data["owner"]).astype(np.int32)

from sklearning.utils.StandardScaler import StandardScaler

data = data.drop(["fuel","seller_type","owner","transmission","name"],axis=1)
sc = StandardScaler()
data = sc.transform(data)

x = data.drop(["selling_price"],axis=1)
y = data["selling_price"]
X = x.to_numpy()
Y = y.to_numpy()