import pandas as pd
import os
print(os.listdir())

from sklearning.utils.LabelEncoder import LabelEncoder


le = LabelEncoder()
print(le.fit_transform(pd.Series(["a","b","b","c"])))
res = le.fit_transform(pd.Series(["a","b","b","c"]))
print(le.inverse_transform(res))