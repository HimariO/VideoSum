import numpy as np
import tensorflow as tf
from dnc.controller import BaseController

"""
A 1-layer LSTM recurrent neural network with 256 hidden units
Note: the state of the LSTM is not saved in a variable becuase we want
the state to reset to zero on every input sequnece
"""

class PostController:

    def __init__(self, input_size, output_size, batch_size=1, cell_num=256):
        """
        PostController will getting memory readvector and controller pre-output as input.
        input size [word_size * readhead + batch_size * dnc_output_size, 1]
        output size is equal to DNC output size.
        """
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size

        with tf.name_scope('post_controller'):
            self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(cell_num)
            self.state = self.lstm_cell.zero_state(self.batch_size, tf.float32)

            # self.output_W = tf.Variable(tf.random_normal([cell_num, output_size], stddev=0.1), name='post_contorler_Wout')

    def network_op(self, pre_output, flat_read_vectors, state):
        X = tf.concat([pre_output, flat_read_vectors], 1)
        X = tf.convert_to_tensor(X)
        lstm_out, new_state = self.lstm_cell(X, state)
        final_out = tf.contrib.layers.fully_connected(lstm_out, num_outputs=self.output_size, trainable=True)
        return final_out, new_state

    def get_state(self):
        return self.state
