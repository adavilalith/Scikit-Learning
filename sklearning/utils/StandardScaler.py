import pandas as pd
class StandardScaler:
    def __init__(self):
        pass
    def transform(self,X):
        if type(X)==pd.Series:   
            mean = X.mean()
            std = X.std()
            scaled = (X - mean)/std
            return scaled 
        if type(X) == pd.DataFrame:
            scaled_df = pd.DataFrame()
            for c in X:
                mean = X[c].mean()
                std = X[c].std() 
                if std==0:
                    scaled_df[c] = 1
                else:
                    scaled_df[c] = (X[c]-mean)/std
            return scaled_df
        else:
            raise Exception("Invalid datatype, use pandas series only")