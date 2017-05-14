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
            self.lstm_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(1300, dropout_keep_prob=0.5)
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
        self.node = 1024

        # self.initializer = tf.random_normal_initializer(stddev=2 / (self.input_size))
        self.initializer = tf.orthogonal_initializer()

        self.lstm_cell = tf.contrib.rnn.LSTMCell(self.node, initializer=self.initializer)
        self.lstm_cell = tf.contrib.rnn.DropoutWrapper(self.lstm_cell, output_keep_prob=0.5)

        if self.layer > 1:
            self.stack_lstm = tf.contrib.rnn.MultiRNNCell([self.lstm_cell] * self.layer)
        else:
            self.stack_lstm = self.lstm_cell
        self.state = self.stack_lstm.zero_state(self.batch_size, tf.float32)

    def network_op(self, X, state):

        with tf.variable_scope('L2_LSTM_Controller'):
            X = tf.convert_to_tensor(X)
            return self.stack_lstm(X, state)

    def get_state(self):
        return self.state

    def update_state(self, new_state):
        return tf.no_op()


class L2NRecurrentController(BaseController):

    def network_vars(self):
        self.layer = 2
        initializer = tf.contrib.layers.xavier_initializer()
        self.lstm_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(256)
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


class L2N512RnnController(L2NRecurrentController):

    def network_vars(self):
        self.layer = 2
        initializer = tf.contrib.layers.xavier_initializer()
        self.lstm_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(768, dropout_keep_prob=0.5)
        # self.lstm_cell = tf.contrib.rnn.LSTMCell(256, initializer=initializer, use_peepholes=True)
        self.stack_lstm = tf.contrib.rnn.MultiRNNCell([self.lstm_cell] * self.layer)
        self.state = self.stack_lstm.zero_state(self.batch_size, tf.float32)


class MemRNNController(L2RecurrentController):
    def initials(self):
        # defining internal weights of the controller
        self.interface_weights = tf.Variable(
            tf.random_normal([self.nn_output_size, self.interface_vector_size], stddev=0.1),
            name='interface_weights'
        )
        self.mem_output_weights = tf.Variable(
            tf.random_normal([self.word_size * self.read_heads, self.output_size], stddev=0.1),
            name='mem_output_weights'
        )

    def process_input(self, X, last_read_vectors, state=None):
        flat_read_vectors = tf.reshape(last_read_vectors, (-1, self.word_size * self.read_heads))
        complete_input = tf.concat([X, flat_read_vectors], 1)
        nn_output, nn_state = None, None

        if self.has_recurrent_nn:
            nn_output, nn_state = self.network_op(complete_input, state)
        else:
            nn_output = self.network_op(complete_input)

        interface = tf.matmul(nn_output, self.interface_weights)
        parsed_interface = self.parse_interface_vector(interface)

        if self.has_recurrent_nn:
            return None, parsed_interface, nn_state
        else:
            return None, parsed_interface

    def final_output(self, pre_output, new_read_vectors):
        """
        We are not using pre_output here, so it will always be None.
        """
        flat_read_vectors = tf.reshape(new_read_vectors, (-1, self.word_size * self.read_heads))

        final_output = tf.matmul(flat_read_vectors, self.mem_output_weights)

        return final_output
