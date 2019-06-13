"""Implementation of GRU+SVM model for MNIST with Dropout"""
import numpy as np
import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data

data = input_data.read_data_sets('/home/darth/GitHub Projects/mnist/data', one_hot=True)

# hyper-parameters
BATCH_SIZE = 256
CELL_SIZE = 256
DROPOUT_P_KEEP = 0.85
EPOCHS = 100
LEARNING_RATE = 1e-3
NUM_CLASSES = 10
SVM_C = 1

# dataset dimension
CHUNK_SIZE = 28
NUM_CHUNKS = 28

CHECKPOINT_PATH = 'checkpoint/'
MODEL_NAME = 'model.ckpt'

LOGS_PATH = 'logs/rnn/'

x = tf.placeholder(dtype=tf.float32, shape=[None, NUM_CHUNKS, CHUNK_SIZE], name='x_input')
y = tf.placeholder(dtype=tf.float32, shape=[None, NUM_CLASSES], name='y_input')
h = tf.placeholder(dtype=tf.float32, shape=[None, CELL_SIZE], name='state')
learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')
p_keep = tf.placeholder(dtype=tf.float32, name='p_keep')


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def recurrent_neural_network(x):
    with tf.name_scope('weights_and_biases'):
        with tf.name_scope('weights'):
            xav_init = tf.contrib.layers.xavier_initializer
            weight = tf.get_variable('weights', shape=[CELL_SIZE, NUM_CLASSES], initializer=xav_init())
            variable_summaries(weight)
        with tf.name_scope('biases'):
            bias = tf.get_variable('biases', initializer=tf.constant(0.1, shape=[NUM_CLASSES]))
            variable_summaries(bias)

    cell = tf.contrib.rnn.GRUCell(CELL_SIZE)
    drop_cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=p_keep)
    
    outputs, states = tf.nn.dynamic_rnn(drop_cell, x, initial_state=h, dtype=tf.float32)

    states = tf.identity(states, name='H')
    hf = tf.transpose(outputs, [1, 0, 2])
    last = tf.gather(hf, int(hf.get_shape()[0]) - 1)
    with tf.name_scope('Wx_plus_b'):
        output = tf.matmul(last, weight) + bias
        tf.summary.histogram('pre-activations', output)

    return output, weight, states


def train_neural_network(x):
    prediction, weight, states = recurrent_neural_network(x)

    with tf.name_scope('loss'):
        regularization_loss = 0.5 * tf.reduce_sum(tf.square(weight))
        hinge_loss = tf.reduce_sum(tf.square(tf.maximum(tf.zeros([BATCH_SIZE, NUM_CLASSES]),
                                                        1 - tf.cast(y, tf.float32) * prediction)))
        with tf.name_scope('loss'):
            cost = regularization_loss + SVM_C * hinge_loss
    tf.summary.scalar('loss', cost)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    with tf.name_scope('accuracy'):
        predicted_class = tf.sign(prediction)
        predicted_class = tf.identity(predicted_class, name='prediction')
        with tf.name_scope('correct_prediction'):
            correct = tf.equal(tf.argmax(predicted_class, 1), tf.argmax(y, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()

    timestamp = str(time.asctime())
    writer = tf.summary.FileWriter(LOGS_PATH + timestamp, graph=tf.get_default_graph())

    saver = tf.train.Saver(max_to_keep=10)

    current_state = np.zeros([BATCH_SIZE, CELL_SIZE])

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)

        checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_PATH)

        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, tf.train.latest_checkpoint(CHECKPOINT_PATH))
        try:
            for epoch in range(EPOCHS):
                epoch_loss = 0
                for _ in range(int(data.train.num_examples / BATCH_SIZE)):
                    epoch_x, epoch_y = data.train.next_batch(BATCH_SIZE)
                    epoch_y[epoch_y == 0] = -1

                    epoch_x = epoch_x.reshape((BATCH_SIZE, NUM_CHUNKS, CHUNK_SIZE))

                    feed_dict = {x: epoch_x, y: epoch_y, h: current_state,
                                 learning_rate: LEARNING_RATE, p_keep: DROPOUT_P_KEEP}

                    summary, _, next_state, c, accuracy_ = sess.run([merged, optimizer, states, cost, accuracy],
                                                                    feed_dict=feed_dict)

                    epoch_loss = c
                    current_state = next_state

                if epoch % 2 == 0:
                    saver.save(sess, CHECKPOINT_PATH + MODEL_NAME, global_step=epoch)
                writer.add_summary(summary, epoch)
                print('Epoch : {} completed out of {}, loss : {}, accuracy : {}'.format(epoch, EPOCHS,
                                                                                        epoch_loss, accuracy_))
        except KeyboardInterrupt:
            print('Training interrupted at {}'.format(epoch))
        finally:
            writer.close()

        saver.save(sess, CHECKPOINT_PATH + MODEL_NAME, global_step=epoch)

        x_ = data.test.images.reshape((-1, NUM_CHUNKS, CHUNK_SIZE))
        y_ = data.test.labels
        y_[y_ == 0] = -1

        accuracy_ = sess.run(accuracy, feed_dict={x: x_, y: y_,
                                                  h: np.zeros([10000, CELL_SIZE]),
                                                                      p_keep: 1.0})
        
        print('Accuracy : {}'.format(accuracy_))
  

if __name__ == '__main__':
    train_neural_network(x)
