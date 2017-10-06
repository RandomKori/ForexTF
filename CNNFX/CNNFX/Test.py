import numpy as np
import CNNModel as rsn
import pandas as pd

LEARNING_RATE=0.0001
EPOCHS=1000

def Read(s):
    dat=pd.read_csv(s,',')
    x=dat[dat.columns[1:]].values
    y=dat[dat.columns[:1]].values
    x=np.array(x)
    y=np.array(y)
    return x,y

def one_hot(y_):
    """
    Function to encode output labels from number indexes.
    E.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    """
    n_values = int(np.max(y_)) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS

x_t,y_t = Read("./TestData/train.csv")
x_e,y_e = Read("./TestData/train.csv")
y_t=one_hot(y_t)
y_t=np.resize(y_t,(y_t.shape[0],38))
y_e=one_hot(y_e)
y_e=np.resize(y_t,(y_e.shape[0],38))
net=rsn.CNN(176,38)
net.k_size=10
net.ftl=7
net.n_layers=9
net.learning_rate=LEARNING_RATE
net.epchs=EPOCHS
net.batch_size=60
net.build_model()
net.build_adam_trainer()
print("Тренировка модели")
net.train(x_t,y_t,x_e,y_e)
print("Тренировка закончена")
input("Нажмите ентер")