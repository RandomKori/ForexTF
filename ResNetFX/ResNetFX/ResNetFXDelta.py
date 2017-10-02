import numpy as np
import ResNetModel as rsn
import pandas as pd

LEARNING_RATE=0.005
EPOCHS=100

def Read(s):
    dt={'s1':np.float64,'s2':np.float64,'s3':np.float64,'s4':np.float64,'s5':np.float64,'s6':np.float64,'s7':np.float64,'s8':np.float64,'s9':np.float64,'s10':np.float64,'s11':np.float64,'s12':np.float64,'s13':np.float64,'s14':np.float64,'s15':np.float64,
        's16':np.float64,'s17':np.float64,'s18':np.float64,'s19':np.float64,'s20':np.float64,'s21':np.float64,'s22':np.float64,'s23':np.float64,'s24':np.float64,'s25':np.float64,'s26':np.float64,'s27':np.float64,'s28':np.float64,'s29':np.float64,'s30':np.float64,
        'l1':np.float64,'l2':np.float64,'l3':np.float64}
    dat=pd.read_csv(s,';',dtype=dt)
    x=dat[dat.columns[:30]].values
    y=dat[dat.columns[30:]].values
    x=np.array(x)
    y=np.array(y)
    return x,y

x_t,y_t = Read("./Data/train.csv")
x_e,y_e = Read("./Data/test.csv")
net=rsn.ResNet(30)
net.k_size=8
net.n_layers=10
net.learning_rate=LEARNING_RATE
net.epchs=EPOCHS
net.batch_size=5000
net.build_model()
net.build_adam_trainer()
print("Тренировка модели")
net.train(x_t,y_t,x_e,y_e)
print("Тренировка закончена")
input("Нажмите ентер")