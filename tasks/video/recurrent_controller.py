import numpy as np
import tensorflow as tf
from dnc.controller import BaseController

"""
A 1-layer LSTM recurrent neural network with 256 hidden units
Note: the state of the LSTM is not saved in a variable becuase we want
the state to reset to zero on every input sequnece
"""

class RecurrentController(BaseController):

    def network_vars(self):
        with tf.variable_scope('LSTM_Controller'):
            self.lstm_cell = tf.contrib.rnn.LSTMCell(256)
            self.state = self.lstm_cell.zero_state(self.batch_size, tf.float32)

    def network_op(self, X, state):
        X = tf.convert_to_tensor(X)
        return self.lstm_cell(X, state)

    def get_state(self):
        return self.state

    def update_state(self, new_state):
        return tf.no_op()


class L2RecurrentController(BaseController):

    def network_vars(self):
        self.layer = 2
        initializer = tf.contrib.layers.xavier_initializer()
        self.lstm_cell = tf.contrib.rnn.LSTMCell(256, initializer=initializer, use_peepholes=True)
        self.stack_lstm = tf.contrib.rnn.MultiRNNCell([self.lstm_cell] * self.layer)
        self.state = self.stack_lstm.zero_state(self.batch_size, tf.float32)

    def network_op(self, X, state):

        with tf.variable_scope('L2_LSTM_Controller'):
            X = tf.convert_to_tensor(X)
            return self.stack_lstm(X, state)

    def get_state(self):
        return self.state

    def update_state(self, new_state):
        return tf.no_op()
