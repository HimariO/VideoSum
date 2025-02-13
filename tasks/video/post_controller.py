import sys
sys.path.append(__file__[:-len("/recurrent_controller.py")])

import numpy as np
import tensorflow as tf
from dnc.controller import BaseController
from tensorflow_toolbox.weight_norm_LSTM import WeightNormLSTMCell


class PostController:

    def __init__(self, input_size, output_size, batch_size=1, cell_num=512, layer=3):
        """
        PostController will getting memory readvector and controller pre-output as input.
        input size [word_size * readhead + batch_size * dnc_output_size, 1]
        output size is equal to DNC output size.
        """
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.layer = layer
        # self.initializer = tf.orthogonal_initializer()
        self.initializer = tf.random_normal_initializer(stddev=0.1)

        self.lstm_cell = WeightNormLSTMCell(cell_num, initializer=self.initializer)
        # self.lstm_cell = tf.contrib.rnn.DropoutWrapper(self.lstm_cell, output_keep_prob=0.5)
        self.lstm_cell = tf.nn.rnn_cell.ResidualWrapper(self.lstm_cell)

        self.final_lstm = WeightNormLSTMCell(cell_num, initializer=self.initializer, num_proj=self.output_size)
        self.final_lstm = tf.nn.rnn_cell.ResidualWrapper(self.final_lstm)

        if self.layer > 1:
            self.stack_lstm = tf.contrib.rnn.MultiRNNCell([self.lstm_cell for _ in range(self.layer - 1)] + [self.final_lstm])
        else:
            self.stack_lstm = self.final_lstm
        self.state = self.stack_lstm.zero_state(self.batch_size, tf.float32)

        # self.output_W = tf.Variable(tf.random_normal([cell_num, output_size], stddev=0.1), name='post_contorler_Wout')

    def network_op(self, final_out, state):
        with tf.variable_scope('post_controller'):
            X = final_out
            X = tf.convert_to_tensor(X)
            lstm_out, new_state = self.stack_lstm(X, state)
            final_out = lstm_out
        return final_out, new_state

    def get_state(self):
        return self.state

    def update_state(self, new_state):
        return tf.no_op()


class DirectPostController:

    def __init__(self, input_size, output_size, batch_size=1, cell_num=512, layer=3):
        """
        DirectPostController will getting [(memory readvector+controller pre-output), videofeature] as input.
        input size [word_size * readhead + batch_size * dnc_output_size, 1]
        output size is equal to DNC output size.
        """
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.layer = layer
        self.cell_num = cell_num
        # self.initializer = tf.orthogonal_initializer()
        self.initializer = tf.random_normal_initializer(stddev=0.1)

        with tf.variable_scope('dirPostController', reuse=True):

            self.lstm_cell = WeightNormLSTMCell(cell_num, initializer=self.initializer)
            # self.lstm_cell = tf.contrib.rnn.DropoutWrapper(self.lstm_cell, output_keep_prob=0.5)
            # self.lstm_cell = tf.nn.rnn_cell.ResidualWrapper(self.lstm_cell)

            self.final_lstm = WeightNormLSTMCell(cell_num, initializer=self.initializer, num_proj=self.output_size)
            # self.final_lstm = tf.nn.rnn_cell.ResidualWrapper(self.final_lstm)

            if self.layer > 1:
                self.stack_lstm = tf.contrib.rnn.MultiRNNCell([self.lstm_cell for _ in range(self.layer - 1)] + [self.final_lstm])
            else:
                self.stack_lstm = self.final_lstm
            self.state = self.stack_lstm.zero_state(self.batch_size, tf.float32)
        # self.stack_lstm = tf.make_template('LSTMCell', self.stack_lstm)

        # self.output_weights = tf.Variable(
        #     tf.random_normal([self.cell_num, self.output_size], stddev=0.1),
        #     name='nn_output_weights'
        # )

    def network_op(self, dnc_out, video_feats, state):
        with tf.variable_scope('dirPostController'):
            X = tf.concat([dnc_out, video_feats], 1)
            X = tf.convert_to_tensor(X)
            lstm_out, new_state = self.stack_lstm(X, state)
            final_out = lstm_out

            return final_out, new_state

    def get_state(self):
        return self.state

    def update_state(self, new_state):
        return tf.no_op()
