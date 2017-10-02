import tensorflow as tf
import numpy as np

class ResNet:
    
    def __init__(self,input_size=45,num_classes=3,layers=20,epochs=100,kernel_size=4):
        self.inp_size=input_size
        self.n_classes=num_classes
        self.n_layers=layers
        self.k_size=kernel_size
        self.epchs=epochs
        self.batch_size=1024
        self.learning_rate=0.01
        self.bn_epsilon=0.001

    def _batch_norm(self,o):
        mean, variance = tf.nn.moments(o, axes=[0, 1, 2])
        beta=tf.Variable(0.0,dtype=tf.float32)
        gamma=tf.Variable(1.0,dtype=tf.float32)
        bn=tf.nn.batch_normalization(o, mean, variance, beta, gamma, self.bn_epsilon)
        return bn

    def build_model(self):
        with tf.variable_scope("Imputs"):
            self.x=tf.placeholder(tf.float32,[None,self.inp_size,1])
            self.y=tf.placeholder(tf.float32,[None,self.n_classes])
        with tf.variable_scope("Layer_inp"):
            output=tf.layers.conv1d(self.x,1,self.k_size,padding="same")
            output=self._batch_norm(output)
            output=tf.layers.conv1d(output,1,self.k_size,padding="same")
            output=self._batch_norm(output)
            output=tf.nn.relu(output)
            k=output
        for i in range(self.n_layers):
            with tf.variable_scope("Layer_{}".format(i)):
                output=tf.layers.conv1d(output,1,self.k_size,padding="same")
                output=self._batch_norm(output)
                output=tf.layers.conv1d(output,1,self.k_size,padding="same")
                output=self._batch_norm(output)
                output=tf.nn.relu(output)
                g=tf.identity(output)
                output=tf.add(k,output)
                k=g
        with tf.variable_scope("Layer_out"):
            output=tf.reshape(output,[tf.shape(output)[0],self.inp_size])
            self.classifier=tf.layers.dense(output,self.n_classes,activation=tf.nn.relu)
        with tf.variable_scope("Metrics"):
            _,self.accuracy = tf.metrics.accuracy(labels=self.y, predictions=self.classifier)
            tf.summary.scalar(name="Accuracy", tensor=self.accuracy)

    def build_mom_trainer(self):
        self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=self.y, logits=self.classifier,label_smoothing=0.1))
        self.train_step = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.5, use_nesterov=True).minimize(loss=self.loss, global_step=tf.train.get_global_step())
        tf.summary.scalar(name="Cross Entropy", tensor=self.loss)

    def train(self,x_train,y_train,x_test,y_test):
        merged = tf.summary.merge_all()
        init_global = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        saver = tf.train.Saver()
        x_train.resize(x_train.shape[0],self.inp_size,1)
        x_test.resize(x_test.shape[0],self.inp_size,1)
        n_batches = int(x_train.shape[0] / self.batch_size)
        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter(logdir="./logs/train/", graph=sess.graph)
            test_writer = tf.summary.FileWriter(logdir="./logs/test/", graph=sess.graph)
            sess.run(fetches=init_global)
            for e in range(1, self.epchs + 1):
                for s in range(n_batches):
                    feed = {self.x: x_train[s*self.batch_size:s*self.batch_size+self.batch_size], self.y: y_train[s*self.batch_size:s*self.batch_size+self.batch_size]}
                    summary,acc = sess.run([merged, self.train_step], feed_dict=feed)
                    train_writer.add_summary(summary, e * n_batches + s)
                summary,acc = sess.run([merged, self.loss],feed_dict={self.x: x_test, self.y: y_test})
                test_writer.add_summary(summary, e)
                loss_train = self.loss.eval(feed_dict={self.x: x_train, self.y: y_train})
                loss_test = self.loss.eval(feed_dict={self.x: x_test, self.y: y_test})
                acc_train = sess.run([self.accuracy],feed_dict={self.x: x_train, self.y: y_train})
                acc_test = sess.run([self.accuracy],feed_dict={self.x: x_test, self.y: y_test})
                print("Эпоха: {0} Ошибка: {1} {3} Ошибка на тестовых данных: {2} {4}".format(e,loss_train,loss_test,1.0-acc_train[0],1.0-acc_test[0]))
                if(loss_train < 0.02):
                    break
            saver.save(sess=sess, save_path="./ResNetFX/ResNetFX")
            rez = sess.run(self.classifier,feed_dict={self.x: x_test})
            for i in range(len(rez)):
                print(rez[i])
