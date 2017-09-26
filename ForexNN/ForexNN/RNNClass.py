import Readers as rd
import numpy as np
import tensorflow as tf

INITIAL_LEARNING_RATE = 0.5
LEARNING_RATE_DECAY_STEPS = 15000
LEARNING_RATE_DECAY_RATE = 0.96
EPOCHS=100
BATCH_SIZE=512
LAYERS=5

def model_rnn(x_t,y_t,x_e,y_e):
    with tf.variable_scope("Inputs"):
        x=tf.placeholder(tf.float32,[None,45,1],"Input")
        y=tf.placeholder(tf.float32,[None,3],"Output")
        training = tf.placeholder_with_default(input=False, shape=None, name="dropout_switch")
    with tf.variable_scope("Net"):
        l_cells=[tf.nn.rnn_cell.BasicRNNCell(45) for _ in range(LAYERS)]       
        cells = tf.nn.rnn_cell.MultiRNNCell(cells=l_cells)
        rnn_output, rnn_state = tf.nn.dynamic_rnn(cell=cells, inputs=x, dtype=tf.float32)
    with tf.variable_scope("predictions"):
        output = rnn_state[-1]
        prediction = tf.layers.dense(inputs=output, units=3, name="prediction")
    with tf.variable_scope("train"):
        global_step = tf.Variable(initial_value=0, trainable=False, name="global_step")
        loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=prediction)
        train_step = tf.train.MomentumOptimizer(learning_rate=0.04,momentum=0.8,use_nesterov=True).minimize(loss=loss, global_step=global_step)
        tf.summary.scalar(name="Cross Entropy", tensor=loss)

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
                feed = {x: x_t[id_batch], y: y_t[id_batch], training: True}
                summary,acc= sess.run([merged, train_step], feed_dict=feed)
                train_writer.add_summary(summary, e*s)
            summary,acc = sess.run([merged, loss],feed_dict={x: x_e, y: y_e, training: False})
            test_writer.add_summary(summary, e)
            loss_train = loss.eval(feed_dict={x: x_t, y: y_t, training: False})
            loss_test = loss.eval(feed_dict={x: x_e, y: y_e, training: False})
            print("Эпоха: {0} Ошибка: {1} Ошибка на тестовых данных: {2}".format(e,loss_train,loss_test))
        saver.save(sess=sess, save_path="./ModelRNNClass/RNNClass")
    return

x_t,y_t=rd.ReadDataClass("./Data/train.csv")
x_t.resize((x_t.shape[0],45,1))
x_e,y_e=rd.ReadDataClass("./Data/test.csv")
x_e.resize((x_e.shape[0],45,1))
print("Тренировка модели")
model_rnn(x_t,y_t,x_e,y_e)
print("Тренировка закончена")
input("Нажмите любую клпвишу")