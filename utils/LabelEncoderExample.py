import pandas as pd
from LabelEncoder import LabelEncoder

print(le.fit_transform(pd.Series(["a","b","b","c"])))
res = le.fit_transform(pd.Series(["a","b","b","c"]))
print(le.inverse_transform(res))