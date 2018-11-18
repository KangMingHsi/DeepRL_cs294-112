import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import tf_util
import pickle

def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.loads(f.read())
        
        return data

class BCModel(object):
    def __init__(self, net_param, input_size, batch_size, action_size, lr=0.5, epoch=1):

        self.net_param = net_param
        self.input_size = input_size
        self.batch_size = batch_size
        self.action_size = action_size
        self.lr = lr
        self.epoch = epoch

        self.state = tf.placeholder(tf.float32, shape=[None, self.input_size], name='state')
        self.action_star = tf.placeholder(tf.float32, shape=[None, self.action_size], name='action_star')

        self.build_model()
        self.build_loss()

    def build_model(self):

        with tf.variable_scope('dense') as scope:

            d1 = tf.nn.relu(tf_util.dense(name='d1', x=self.state, weight_init=tf_util.normc_initializer(), size=self.net_param['d1']))
            d2 = tf.nn.relu(tf_util.dense(name='d2', x=d1, weight_init=tf_util.normc_initializer(), size=self.net_param['d2']))
            d3 = tf.nn.relu(tf_util.dense(name='d3', x=d2, weight_init=tf_util.normc_initializer(), size=self.net_param['d3']))

            self.action = tf.tanh(tf_util.dense(name='out', x=d3, weight_init=tf_util.normc_initializer(), size=self.action_size))

    def build_loss(self):

        self.loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.action_star, predictions=self.action))
        self.opt = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)


    def fit(self, inputs, outputs, sess):
        self.saver = tf.train.Saver()
        for epoch in range(self.epoch):
            losses = []
            inputs, outputs = shuffle(inputs, outputs, random_state = 0)
            for batch_step in range(inputs.shape[0] // self.batch_size):

                _from = int(batch_step*self.batch_size)
                _to = int((batch_step+1)*self.batch_size) if int((batch_step+1)*self.batch_size) < inputs.shape[0] else inputs.shape[0]

                batch_state = inputs[_from:_to]
                batch_action_star = outputs[_from:_to]

                loss, _ = sess.run([self.loss, self.opt],                                      feed_dict={self.state:batch_state, self.action_star:batch_action_star})

                losses.append(loss)
            #print(np.mean(losses))
        #self.saver.save(sess, 'model/ant-v2.ckpt')
    def predict(self, inputs, sess):
        return sess.run(self.action, feed_dict={self.state:inputs})
    def save(self, fname, sess):
        self.saver.save(sess, 'model/' + fname + '.ckpt')
    def load(self, fname, sess):
        self.saver.restore(sess, 'model/' + fname + '.ckpt')
