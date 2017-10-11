import numpy as np
import pandas as pd
from keras.models import Sequential
import keras as kr
import tensorflow as tf
from keras import backend as K

def Read(s):
    dt={'s1':np.float64,'s2':np.float64,'s3':np.float64,'s4':np.float64,'s5':np.float64,'s6':np.float64,'s7':np.float64,'s8':np.float64,'s9':np.float64,'s10':np.float64,
        's11':np.float64,'s12':np.float64,'s13':np.float64,'s14':np.float64,'s15':np.float64,'s16':np.float64,'s17':np.float64,'s18':np.float64,'s19':np.float64,'s20':np.float64,
        's21':np.float64,'s22':np.float64,'s23':np.float64,'s24':np.float64,'s25':np.float64,'s26':np.float64,'s27':np.float64,'s28':np.float64,'s29':np.float64,'s30':np.float64,
        'l1':np.float64,'l2':np.float64,'l3':np.float64}
    dat=pd.read_csv(s,';',dtype=dt)
    x=dat[dat.columns[:30]].values
    y=dat[dat.columns[30:]].values
    x=np.array(x)
    y=np.array(y)
    return x,y

x_t,y_t = Read("./Data/train.csv")
x_e,y_e = Read("./Data/test.csv")
x_t.resize(x_t.shape[0],10,3)
x_e.resize(x_e.shape[0],10,3)
tensorboard=kr.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=512, write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
model = Sequential()
model.add(kr.layers.LSTM(units=90, return_sequences=True, input_shape=(10, 3)))
for i in range(2):
    model.add(kr.layers.LSTM(units=90,return_sequences=True))
model.add(kr.layers.LSTM(32))
model.add(kr.layers.Dense(units=3))
model.add(kr.layers.Activation('softmax'))
model.compile(loss=kr.losses.categorical_crossentropy,
              optimizer=kr.optimizers.SGD(lr=0.0001, momentum=0.9, nesterov=True),metrics=['accuracy'])

saver = tf.train.Saver()
sess = tf.Session()
K.set_session(sess)
model.fit(x_t, y_t, epochs=1000, batch_size=512, verbose=1, validation_data=(x_e, y_e), callbacks=[tensorboard])
saver.save(sess=sess, save_path="./LTSMModel/LTSMModel")
loss_and_metrics = model.evaluate(x_e, y_e, batch_size=512)
classes = model.predict(x_e, batch_size=512)
for i in range(len(classes)):
    print(classes[i])
sess.close()
input("Нажмите ентер")
