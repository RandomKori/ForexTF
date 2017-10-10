import numpy as np
import pandas as pd
from keras.models import Sequential
import keras as kr
import tensorflow as tf
from keras import backend as K

def Read(s):
    dt={'s1':np.float64,'s2':np.float64,'s3':np.float64,'s4':np.float64,'s5':np.float64,'s6':np.float64,'s7':np.float64,'s8':np.float64,'s9':np.float64,'s10':np.float64,'s11':np.float64,'s12':np.float64,'s13':np.float64,'s14':np.float64,'s15':np.float64,
        's16':np.float64,'s17':np.float64,'s18':np.float64,'s19':np.float64,'s20':np.float64,'s21':np.float64,'s22':np.float64,'s23':np.float64,'s24':np.float64,'s25':np.float64,'s26':np.float64,'s27':np.float64,'s28':np.float64,'s29':np.float64,'s30':np.float64,
        's31':np.float64,'s32':np.float64,'s33':np.float64,'s34':np.float64,'s35':np.float64,'s36':np.float64,'s37':np.float64,'s38':np.float64,'s39':np.float64,'s40':np.float64,'s41':np.float64,'s42':np.float64,'s43':np.float64,'s44':np.float64,'s45':np.float64,
        'l1':np.float64,'l2':np.float64,'l3':np.float64}
    dat=pd.read_csv(s,';',dtype=dt)
    x=dat[dat.columns[:45]].values
    y=dat[dat.columns[45:]].values
    x=np.array(x)
    y=np.array(y)
    return x,y

x_t,y_t = Read("./Data/train.csv")
x_e,y_e = Read("./Data/test.csv")
model = Sequential()
model.add(kr.layers.Dense(units=90, input_dim=45))
model.add(kr.layers.Activation('tanh'))
for i in range(4):
    model.add(kr.layers.Dense(units=90))
    model.add(kr.layers.Activation('tanh'))
model.add(kr.layers.Dense(units=3))
model.add(kr.layers.Activation('softmax'))
model.compile(loss=kr.losses.categorical_crossentropy,
              optimizer=kr.optimizers.SGD(lr=0.0001, momentum=0.9, nesterov=True),metrics=['accuracy'])

saver = tf.train.Saver()
sess = tf.Session()
K.set_session(sess)
model.fit(x_t, y_t, epochs=100, batch_size=512)
saver.save(sess=sess, save_path="./TestModel/TestModel")
loss_and_metrics = model.evaluate(x_e, y_e, batch_size=512)
classes = model.predict(x_e, batch_size=512)
for i in range(len(classes)):
    print(classes[i])
sess.close()
input("Нажмите ентер")
