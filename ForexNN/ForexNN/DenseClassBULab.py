import Readers as rd
import numpy as np
import tensorflow as tf

INITIAL_LEARNING_RATE = 0.05
LEARNING_RATE_DECAY_RATE = 0.01
EPOCHS=5000
BATCH_SIZE=1024
LAYERS=7

def model_rnn(x_t,y_t,z_t,x_e,y_e,z_e):
    with tf.variable_scope("Inputs"):
        x=tf.placeholder(tf.float32,[None,30],"Input")
        y=tf.placeholder(tf.float32,[None,30],"Output")
        z=tf.placeholder(tf.float32,[None,3],"ClassesOut")

    with tf.variable_scope("Net"):
        output = tf.layers.dense(inputs=x, units=70, activation=tf.nn.sigmoid, name="layer_inp")
        for i in range(LAYERS):
            output = tf.layers.dense(inputs=output, units=70, activation=tf.nn.sigmoid, name="layer_"+"{}".format(i))
        classifier = tf.layers.dense(inputs=output, units=4, activation=tf.nn.sigmoid, name="classifier")
        output = tf.layers.dense(inputs=classifier, units=70, activation=tf.nn.sigmoid, name="layer_inpo")
        for i in range(LAYERS):
            output = tf.layers.dense(inputs=output, units=70, activation=tf.nn.sigmoid, name="layer_o_"+"{}".format(i))
        prediction = tf.layers.dense(inputs=output, units=30, activation=tf.nn.sigmoid, name="prediction")

    with tf.variable_scope("Net1"):
        output = tf.layers.dense(inputs=classifier, units=10, activation=tf.nn.sigmoid, name="layer_inp1")
        for i in range(LAYERS):
            output = tf.layers.dense(inputs=output, units=10, activation=tf.nn.sigmoid, name="layer1_"+"{}".format(i))
        classifierlab = tf.layers.dense(inputs=output, units=3, activation=tf.nn.sigmoid, name="classifierlab")

    with tf.variable_scope("train"):
        loss = tf.losses.mean_squared_error(labels=y, predictions=prediction)
        train_step = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.2, use_nesterov=True, name='Momentum').minimize(loss=loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Net'), global_step=tf.train.get_global_step())
        tf.summary.scalar(name="MSE", tensor=loss)

    with tf.variable_scope("train1"):
        loss1 =  tf.losses.softmax_cross_entropy(onehot_labels=z, logits=prediction,reduction=tf.losses.Reduction.MEAN)
        train_step1 = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.2, use_nesterov=True, name='MomentumLab').minimize(loss=loss1,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Net1'), global_step=tf.train.get_global_step())
        tf.summary.scalar(name="Class", tensor=loss1)

    idx = list(range(x_t.shape[0]))
    n_batches = int(np.ceil(len(idx) / BATCH_SIZE))
    merged = tf.summary.merge_all()
    init_global = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(logdir="./logs/train/", graph=sess.graph)
        test_writer = tf.summary.FileWriter(logdir="./logs/test/", graph=sess.graph)
        sess.run(fetches=init_global)

        for e in range(1, EPOCHS + 1):
            np.random.shuffle(idx)
            batch_generator = (idx[i * BATCH_SIZE:(1 + i) * BATCH_SIZE] for i in range(n_batches))
            for s in range(n_batches):
                id_batch = next(batch_generator)
                feed = {x: x_t[id_batch], y: y_t[id_batch], z: z_t[id_batch]}
                summary,acc= sess.run([merged, train_step], feed_dict=feed)
                train_writer.add_summary(summary, e*n_batches+s)
            summary,acc = sess.run([merged, loss],feed_dict={x: x_e, y: y_e, z: z_e})
            test_writer.add_summary(summary, e)
            loss_train = loss.eval(feed_dict={x: x_t, y: y_t})
            loss_test = loss.eval(feed_dict={x: x_e, y: y_e})
            print("Эпоха: {0} Ошибка: {1} Ошибка на тестовых данных: {2}".format(e,loss_train,loss_test))
            if(loss_train<1.0E-7): 
                break

        for e in range(1, EPOCHS + 1):
            np.random.shuffle(idx)
            batch_generator = (idx[i * BATCH_SIZE:(1 + i) * BATCH_SIZE] for i in range(n_batches))
            for s in range(n_batches):
                id_batch = next(batch_generator)
                feed = {x: x_t[id_batch], y: y_t[id_batch], z: z_t[id_batch]}
                summary,acc= sess.run([merged, train_step1], feed_dict=feed)
                train_writer.add_summary(summary, e*n_batches+s)
            summary,acc = sess.run([merged, loss1],feed_dict={x: x_e, y: y_e, z: z_e})
            test_writer.add_summary(summary, e)
            loss_train1 = loss1.eval(feed_dict={x: x_t, z: z_t})
            loss_test1 = loss1.eval(feed_dict={x: x_e, z: z_e})
            print("Эпоха: {0} Ошибка: {1} Ошибка на тестовых данных: {2}".format(e,loss_train1,loss_test1))
            if(loss_train1<0.01): 
                break
        saver.save(sess=sess, save_path="./ModelDenseClassBU/DenseClassBU")
        rez=sess.run(classifierlab,feed_dict={x: x_e})
        for i in range(len(rez)):
            print(rez[i])
    return

x_t,y_t,z_t=rd.ReadDataClassBULab("./Data/train.csv")
#x_t.resize((x_t.shape[0],15,3))
x_e,y_e,z_e=rd.ReadDataClassBULab("./Data/test.csv")
#x_e.resize((x_e.shape[0],15,3))
print("Тренировка модели")
model_rnn(x_t,y_t,z_t,x_e,y_e,z_e)
print("Тренировка закончена")
input("Нажмите любую клпвишу")