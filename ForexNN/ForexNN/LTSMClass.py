import Readers as rd
import numpy as np
import tensorflow as tf

LEARNING_RATE = 0.0001
LEARNING_RATE_DECAY_RATE = 0.96
EPOCHS = 1000
BATCH_SIZE = 5000
LAYERS = 5

def model_rnn(x_t,y_t,x_e,y_e):
    with tf.variable_scope("Inputs"):
        x = tf.placeholder(tf.float32,[None,10,3],"Input")
        y = tf.placeholder(tf.float32,[None,3],"Output")
        

    with tf.variable_scope("Net"):
        l_cells = [tf.nn.rnn_cell.BasicLSTMCell(9) for _ in range(10)]
        rnn_cells = tf.nn.rnn_cell.MultiRNNCell(cells=l_cells)
        output, state = tf.nn.dynamic_rnn(rnn_cells,x,dtype=tf.float32, scope="LTSM_l_inp")  
        for i in range(LAYERS):
            l_cells = [tf.nn.rnn_cell.BasicLSTMCell(9) for _ in range(10)]
            rnn_cells = tf.nn.rnn_cell.MultiRNNCell(cells=l_cells)
            output, state = tf.nn.dynamic_rnn(rnn_cells,output,dtype=tf.float32, scope="LTSM_l_" + "{}".format(i))
        
    with tf.variable_scope("predictions"):
        output = output[:,0]
        prediction = tf.layers.dense(inputs=output, units=3, activation=tf.nn.relu, name="prediction")

    with tf.variable_scope("train"):
        global_step = tf.Variable(initial_value=0, trainable=False, name="global_step")
        loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=prediction,reduction=tf.losses.Reduction.MEAN)
        train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss=loss)
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
                summary,acc = sess.run([merged, train_step], feed_dict=feed)
                train_writer.add_summary(summary, e * n_batches + s)
            summary,acc = sess.run([merged, loss],feed_dict={x: x_e, y: y_e})
            test_writer.add_summary(summary, e)
            loss_train = loss.eval(feed_dict={x: x_t, y: y_t})
            loss_test = loss.eval(feed_dict={x: x_e, y: y_e})
            acc_train = sess.run([accuracy],feed_dict={x: x_t, y: y_t})
            acc_test = sess.run([accuracy],feed_dict={x: x_e, y: y_e})
            print("Эпоха: {0} Ошибка: {1} {3}% Ошибка на тестовых данных: {2} {4}%".format(e,loss_train,loss_test,100.0-acc_train[0],100.0-acc_test[0]))
            if(loss_train < 0.02):
                break
        saver.save(sess=sess, save_path="./ModelRNNClass/RNNClass")
        rez = sess.run(prediction,feed_dict={x: x_e})
        for i in range(len(rez)):
            print(rez[i])
    return

x_t,y_t = rd.ReadDataClassLTSM("./Data/train.csv")
x_t.resize((x_t.shape[0],10,3))
x_e,y_e = rd.ReadDataClassLTSM("./Data/test.csv")
x_e.resize((x_e.shape[0],10,3))
print("Тренировка модели")
model_rnn(x_t,y_t,x_e,y_e)
print("Тренировка закончена")
input("Нажмите любую клпвишу")