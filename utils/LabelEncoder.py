import numpy as np
import pandas as pd
class LabelEncoder:
    def __init__(self):
        self.mapping={}
        self.basekey=0

    def fit_transform(self,data):
        transformed_data = data.copy()
        d={}
        if type(data)==pd.Series:
            for i in range(len(data)):
                val=data[i]
                if val in d:
                    transformed_data.at[i] = d[val]
                else:
                    d[val]=self.basekey
                    self.mapping[self.basekey]=val
                    self.basekey+=1
                    transformed_data.at[i] = d[val]
        else:
            raise Exception("unsupported data format")
        return transformed_data
    
    def inverse_transform(self,data):
        transformed_data = data.copy()
        if type(data)==pd.Series:
            for i in range(len(data)):
                val=data[i]
                transformed_data.at[i] = self.mapping[val]
        else:
            raise Exception("unsupported data format")
        return transformed_data
