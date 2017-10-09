import Readers as rd
import numpy as np
import tensorflow as tf

LEARNING_RATE = 0.0001
EPOCHS = 100
BATCH_SIZE = 1024
LAYERS = 5

def model_rnn(x_t,y_t,x_e,y_e):
    with tf.variable_scope("Inputs"):
        x = tf.placeholder(tf.float32,[None,10,3],"Input")
        y = tf.placeholder(tf.float32,[None,3],"Output")
        

    with tf.variable_scope("Net"):
        #norm=tf.nn.l2_normalize(x,2,name="norm")
        l_cells = [tf.nn.rnn_cell.BasicLSTMCell(3) for _ in range(10)]
        rnn_cells = tf.nn.rnn_cell.MultiRNNCell(cells=l_cells)
        output, state = tf.nn.dynamic_rnn(rnn_cells,x,dtype=tf.float32, scope="LTSM_l_inp")  
        for i in range(LAYERS):
            l_cells = [tf.nn.rnn_cell.BasicLSTMCell(3) for _ in range(10)]
            rnn_cells = tf.nn.rnn_cell.MultiRNNCell(cells=l_cells)
            output, state = tf.nn.dynamic_rnn(rnn_cells,output,dtype=tf.float32, scope="LTSM_l_" + "{}".format(i))
        
    with tf.variable_scope("predictions"):
        output=tf.reshape(output,[tf.shape(output)[0],30])
        prediction = tf.layers.dense(inputs=output, units=3, activation=None, name="prediction")
        classes=tf.nn.softmax(prediction)

    with tf.variable_scope("train"):
        global_step = tf.Variable(initial_value=0, trainable=False, name="global_step")
        loss = tf.losses.softmax_cross_entropy(onehot_labels=y,  logits=prediction)
        train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss=loss, global_step=tf.train.get_global_step())
        tf.summary.scalar(name="Cross Entropy", tensor=loss)

    with tf.variable_scope("Metrics"):
        pred=tf.round(classes)
        lab=tf.cast(y, tf.int32)
        pred=tf.cast(pred, tf.int32)
        accurasy=tf.contrib.metrics.accuracy(labels = lab, predictions = pred)
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
        saver.save(sess=sess, save_path="./ModelRNNClass/RNNClass")
        rez = sess.run(classes,feed_dict={x: x_e})
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