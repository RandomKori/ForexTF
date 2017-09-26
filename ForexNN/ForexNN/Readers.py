import pandas as pd
import numpy as np

def ReadDataClass(s):
    dat=pd.read_csv(s,';')
    x=dat[dat.columns[:45]].values
    y=dat[dat.columns[45:]].values
    x=np.array(x)
    y=np.array(y)
    return x,y