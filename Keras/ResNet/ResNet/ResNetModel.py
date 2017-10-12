import keras
from keras import backend as K
import tensorflow as tf

class ResNet:

    def __init__(self,input_size=45,num_classes=3,layers=20,epochs=100,kernel_size=4):
        self.inp_size=input_size
        self.num_classes=num_classes
        self.n_layers=layers
        self.k_size=kernel_size
        self.ftl=7
        self.epchs=epochs
        self.b_size=1024
        self.learning_rate=0.01
        self.bn_epsilon=0.001
        self.erly_stop=0.01
        self.model=None

    def _con(self,x):
        cn=keras.layers.Conv1D(self.ftl,self.k_size,padding='same')(x)
        cn=keras.layers.Conv1D(self.ftl,self.k_size,padding='same')(cn)
        cn=keras.layers.Activation('relu')(cn)
        cn=keras.layers.BatchNormalization()(cn)
        return cn
    
    def build_resnet(self):
        x=keras.Input(shape=(self.inp_size,1))
        layer=self._con(x)
        f=layer
        for i in range(self.n_layers):
            layer=self._con(f)
            l=layer
            layer=keras.layers.add([f,layer])
            f=l
        y=keras.layers.AveragePooling1D()(layer)
        y=keras.layers.Flatten()(y)
        out=keras.layers.Dense(units=self.num_classes,activation='sigmoid')(y)
        self.model=keras.models.Model(x,out)
        return

    def train(self,x_train,y_train,x_test,y_test):
        self.build_resnet()
        tensorboard=keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=self.b_size, write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
        saver = tf.train.Saver()
        sess = tf.Session()
        K.set_session(sess) 
        optimizer = keras.optimizers.RMSprop(lr=self.learning_rate)
        self.model.compile(loss='binary_crossentropy',
                    optimizer=optimizer,
                     metrics=['accuracy'])
        
        hist = self.model.fit(x_train, y_train, batch_size=self.b_size, epochs=self.epchs,
                verbose=1, validation_data=(x_test, y_test), callbacks = [tensorboard])
        saver.save(sess=sess, save_path="./ResNetModel/ResNetModel")
        classes = self.model.predict(x_test, batch_size=self.b_size)
        for i in range(len(classes)):
            print(classes[i])
        sess.close()
