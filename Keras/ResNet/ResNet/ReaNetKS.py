import keras
import ResNetModel
import numpy as np
import pandas as pd

def readucr(filename):
    dt={'s1':np.float64,'s2':np.float64,'s3':np.float64,'s4':np.float64,'s5':np.float64,'s6':np.float64,'s7':np.float64,'s8':np.float64,'s9':np.float64,'s10':np.float64,'s11':np.float64,'s12':np.float64,'s13':np.float64,'s14':np.float64,'s15':np.float64,
        's16':np.float64,'s17':np.float64,'s18':np.float64,'s19':np.float64,'s20':np.float64,'s21':np.float64,'s22':np.float64,'s23':np.float64,'s24':np.float64,'s25':np.float64,'s26':np.float64,'s27':np.float64,'s28':np.float64,'s29':np.float64,'s30':np.float64,
        's31':np.float64,'s32':np.float64,'s33':np.float64,'s34':np.float64,'s35':np.float64,'s36':np.float64,'s37':np.float64,'s38':np.float64,'s39':np.float64,'s40':np.float64,'s41':np.float64,'s42':np.float64,'s43':np.float64,'s44':np.float64,'s45':np.float64,
        'l1':np.float64}
    dat=pd.read_csv(filename,';',dtype=dt)
    x=dat[dat.columns[:45]].values
    y=dat[dat.columns[45:]].values
    x=np.array(x)
    y=np.array(y)
    return x,y

modelr=ResNetModel.ResNet()
x_train, y_train = readucr("./Data/train.csv")
x_test, y_test = readucr("./Data/test.csv")
modelr.learning_rate=0.0001
modelr.num_classes = 9
modelr.b_size = 128
modelr.epchs=1500
modelr.ftl=18
modelr.k_size=8
modelr.n_layers=10     
x_train.resize(x_train.shape[0],45,1)
x_test.resize(x_test.shape[0],45,1)
y_train=keras.utils.to_categorical(y_train,9)
y_test=keras.utils.to_categorical(y_test,9)
modelr.train(x_train,y_train,x_test,y_test)
input("Нажмите ентер")
