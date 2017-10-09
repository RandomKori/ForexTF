import Readers as rd
import numpy as np
import tensorflow as tf

LEARNING_RATE = 0.0001
EPOCHS=1000
BATCH_SIZE=1024
LAYERS=10

def model_rnn(x_t,y_t,x_e,y_e):
    with tf.variable_scope("Inputs"):
        x=tf.placeholder(tf.float32,[None,45],"Input")
        y=tf.placeholder(tf.float32,[None,2],"Output")
        
    with tf.variable_scope("Net"):
        norm=tf.nn.l2_normalize(x,1,name="norm")
        output = tf.layers.dense(inputs=norm, units=90,activation=tf.nn.tanh, name="layer_inp")
        output=tf.contrib.layers.batch_norm(output, center=True, scale=True)
        for i in range(LAYERS):
            output = tf.layers.dense(inputs=output, units=90,activation=tf.nn.tanh, name="layer_"+"{}".format(i))
            output=tf.contrib.layers.batch_norm(output, center=True, scale=True)
        
    with tf.variable_scope("predictions"):
        prediction = tf.layers.dense(inputs=output, units=2, activation=None, name="prediction")
        classes=tf.nn.sigmoid(prediction,name="Classes")

    with tf.variable_scope("train"):
        loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=y, logits=prediction)
        train_step = tf.train.MomentumOptimizer(learning_rate=LEARNING_RATE, momentum=0.5, use_nesterov=True).minimize(loss=loss, global_step=tf.train.get_global_step())
        tf.summary.scalar(name="Cross Entropy", tensor=loss)

    with tf.variable_scope("Metrics"):
        _,accurasy=tf.contrib.metrics.streaming_auc(labels = y, predictions = classes)
        tf.summary.scalar(name="Accuracy", tensor=accurasy)

    idx = list(range(x_t.shape[0]))
    n_batches = int(np.ceil(len(idx) / BATCH_SIZE))
    merged = tf.summary.merge_all()
    init_global = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(logdir="./logs/train/", graph=sess.graph)
        test_writer = tf.summary.FileWriter(logdir="./logs/test/", graph=sess.graph)
        sess.run(fetches=init_global)
        sess.run(tf.initialize_local_variables())
        for e in range(1, EPOCHS + 1):
                for s in range(n_batches):
                    feed = {x: x_t[s*BATCH_SIZE:s*BATCH_SIZE+BATCH_SIZE], y: y_t[s*BATCH_SIZE:s*BATCH_SIZE+BATCH_SIZE]}
                    acc = sess.run([train_step], feed_dict=feed)
                summary_train,loss_train,acc_train = sess.run([merged, loss, accurasy],feed_dict={x: x_t, y: y_t})
                train_writer.add_summary(summary_train, e)
                summary_test,loss_test,acc_test = sess.run([merged, loss, accurasy],feed_dict={x: x_e, y: y_e})
                test_writer.add_summary(summary_test, e)
                print("Эпоха: {0} Ошибка: {1} {3} Ошибка на тестовых данных: {2} {4}".format(e,loss_train,loss_test,acc_train,acc_test))
                if(loss_train<0.01): 
                    break
        saver.save(sess=sess, save_path="./ModelDenseClass/DenseClass")
        rez=sess.run(classes,feed_dict={x: x_e})
        for i in range(len(rez)):
            print(rez[i])
    return

x_t,y_t=rd.ReadDataClass2L("./Data/train.csv")
#x_t.resize((x_t.shape[0],15,3))
x_e,y_e=rd.ReadDataClass2L("./Data/test.csv")
#x_e.resize((x_e.shape[0],15,3))
print("Тренировка модели")
model_rnn(x_t,y_t,x_e,y_e)
print("Тренировка закончена")
input("Нажмите любую клпвишу")