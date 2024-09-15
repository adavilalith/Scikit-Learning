import pandas as pd
import numpy as np

num_rows = 10
num_columns = 5

data = np.random.rand(num_rows, num_columns)*10

column_names = [f'Feature_{i}' for i in range(num_columns)]

df = pd.DataFrame(data, columns=column_names)

from sklearning.utils import StandardScaler 
sc = StandardScaler()
print(df.head())
scaled_df = sc.transform(df)
print(scaled_df.head())