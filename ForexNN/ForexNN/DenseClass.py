import Readers as rd
import numpy as np
import tensorflow as tf

LEARNING_RATE = 0.1
LEARNING_RATE_DECAY_RATE = 0.001
EPOCHS=5000
BATCH_SIZE=5000
LAYERS=10

def model_rnn(x_t,y_t,x_e,y_e):
    with tf.variable_scope("Inputs"):
        x=tf.placeholder(tf.float32,[None,45],"Input")
        y=tf.placeholder(tf.float32,[None,3],"Output")
        
    with tf.variable_scope("Net"):
        norm=tf.nn.l2_normalize(x,1,name="norm")
        output = tf.layers.dense(inputs=norm, units=128,activation=tf.nn.sigmoid, name="layer_inp")
        for i in range(LAYERS):
            output = tf.layers.dense(inputs=output, units=128,activation=tf.nn.sigmoid, name="layer_"+"{}".format(i))
        
    with tf.variable_scope("predictions"):
        prediction = tf.layers.dense(inputs=output, units=3, activation=tf.nn.relu, name="prediction")

    with tf.variable_scope("train"):
        loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=y, logits=prediction,label_smoothing=0.1))
        train_step = tf.train.MomentumOptimizer(learning_rate=LEARNING_RATE, momentum=0.5, use_nesterov=True).minimize(loss=loss, global_step=tf.train.get_global_step())
        _,accuracy = tf.metrics.accuracy(labels=y, predictions=prediction)
        tf.summary.scalar(name="Cross Entropy", tensor=loss)
        tf.summary.scalar(name="Accuracy", tensor=accuracy)

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
            #np.random.shuffle(idx)
            batch_generator = (idx[i * BATCH_SIZE:(1 + i) * BATCH_SIZE] for i in range(n_batches))
            for s in range(n_batches):
                id_batch = next(batch_generator)
                feed = {x: x_t[id_batch], y: y_t[id_batch]}
                summary,acc= sess.run([merged, train_step], feed_dict=feed)
                train_writer.add_summary(summary, e * n_batches + s)
            summary,acc = sess.run([merged, loss],feed_dict={x: x_e, y: y_e})
            test_writer.add_summary(summary, e*n_batches+s)
            loss_train = loss.eval(feed_dict={x: x_t, y: y_t})
            loss_test = loss.eval(feed_dict={x: x_e, y: y_e})
            acc_train = sess.run([accuracy],feed_dict={x: x_t, y: y_t})
            acc_test = sess.run([accuracy],feed_dict={x: x_e, y: y_e})
            print("Эпоха: {0} Ошибка: {1} {3}% Ошибка на тестовых данных: {2} {4}%".format(e,loss_train,loss_test,acc_train[0],acc_test[0]))
            if(loss_train<0.01): 
                break
        saver.save(sess=sess, save_path="./ModelDenseClass/DenseClass")
        rez=sess.run(prediction,feed_dict={x: x_e})
        for i in range(len(rez)):
            print(rez[i])
    return

x_t,y_t=rd.ReadDataClass("./Data/train.csv")
#x_t.resize((x_t.shape[0],15,3))
x_e,y_e=rd.ReadDataClass("./Data/test.csv")
#x_e.resize((x_e.shape[0],15,3))
print("Тренировка модели")
model_rnn(x_t,y_t,x_e,y_e)
print("Тренировка закончена")
input("Нажмите любую клпвишу")