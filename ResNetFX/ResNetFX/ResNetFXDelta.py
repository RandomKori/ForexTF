import numpy as np
import ResNetModel as rsn
import pandas as pd

LEARNING_RATE=0.0001
EPOCHS=1000

def Read(s):
    dt={'s1':np.float32,'s2':np.float32,'s3':np.float32,'s4':np.float32,'s5':np.float32,'s6':np.float32,'s7':np.float32,'s8':np.float32,'s9':np.float32,'s10':np.float32,'s11':np.float32,'s12':np.float32,'s13':np.float32,'s14':np.float32,'s15':np.float32,
        's16':np.float32,'s17':np.float32,'s18':np.float32,'s19':np.float32,'s20':np.float32,'s21':np.float32,'s22':np.float32,'s23':np.float32,'s24':np.float32,'s25':np.float32,'s26':np.float32,'s27':np.float32,'s28':np.float32,'s29':np.float32,'s30':np.float32,
        'l1':np.float32,'l2':np.float32,'l3':np.float32}
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
net.ftl=15
net.n_layers=10
net.learning_rate=LEARNING_RATE
net.epchs=EPOCHS
net.batch_size=512
net.build_model()
net.build_adam_trainer()
print("Тренировка модели")
net.train(x_t,y_t,x_e,y_e)
print("Тренировка закончена")
input("Нажмите ентер")