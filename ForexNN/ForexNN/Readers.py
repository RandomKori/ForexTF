import pandas as pd

def ReadDataClass(s):
    dat=pd.read_csv(s,';')
    x=dat[dat.columns[1:45]].values
    y=dat[dat.columns[46:]].values
    return x,y