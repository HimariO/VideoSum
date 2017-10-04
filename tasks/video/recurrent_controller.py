import sys
sys.path.append(__file__[:-len("/recurrent_controller.py")])

import math
import numpy as np
import tensorflow as tf
from dnc.controller import BaseController
from SELU import *
from tensorflow_toolbox.weight_norm_LSTM import WeightNormLSTMCell

"""
A 1-layer LSTM recurrent neural network with 256 hidden units
Note: the state of the LSTM is not saved in a variable becuase we want
the state to reset to zero on every input sequnece
"""


class RecurrentController(BaseController):

    def network_vars(self):
        self.lstm_cell = []
        for _ in range(2):
            self.lstm_cell.append(tf.contrib.rnn.LayerNormBasicLSTMCell(1024, dropout_keep_prob=0.7, ))

        self.stack_lstm = tf.contrib.rnn.MultiRNNCell(self.lstm_cell)

        self.state = self.stack_lstm.zero_state(self.batch_size, tf.float32)
        self.stack_lstm = tf.make_template('LSTMCell', self.stack_lstm)

    def network_op(self, X, state):
        X = tf.convert_to_tensor(X)
        return self.stack_lstm(X, state)

    def get_state(self):
        return self.state

    def update_state(self, new_state):
        return tf.no_op()


class L2RecurrentController(BaseController):

    def network_vars(self):
        self.layer = 3
        self.node = 1024

        # self.initializer = tf.random_normal_initializer(stddev=0.1)
        self.initializer = tf.random_normal_initializer(stddev=2 / (self.input_size + self.output_size))
        # self.initializer = tf.orthogonal_initializer()``

        self.lstm_cell = []

        for _ in range(self.layer):
            lstm_cell = WeightNormLSTMCell(
                self.node,
                initializer=self.initializer,
                # state_is_tuple=True,
                activation=selu,
            )
            # lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=0.7)
            if _ > 0:
                lstm_cell = tf.nn.rnn_cell.ResidualWrapper(lstm_cell)
            self.lstm_cell.append(lstm_cell)

        if self.layer > 1:
            self.stack_lstm = tf.contrib.rnn.MultiRNNCell(self.lstm_cell, state_is_tuple=True)
        else:
            self.stack_lstm = self.lstm_cell[0]

        self.state = self.stack_lstm.zero_state(self.batch_size, tf.float32)
        # self.stack_lstm = tf.make_template('LSTMCell', self.stack_lstm)

    def network_op(self, X, state):
        X = tf.convert_to_tensor(X)
        Y, new_state = self.stack_lstm(X, state)

        # Y = tf.contrib.layers.layer_norm(Y)
        return Y, new_state

    def get_state(self):
        return self.state

    def update_state(self, new_state):
        return tf.no_op()


class MemRNNController(L2RecurrentController):
    """
    Controller that only output memory read_vector,
    should be use with direct controller in DNCDuo.
    """

    def process_input(self, X, last_read_vectors, state_mem=None, state_pred=None):
        flat_read_vectors = tf.reshape(last_read_vectors, (-1, self.word_size * self.read_heads))
        complete_input = tf.concat([X, flat_read_vectors], 1)
        nn_output_pred, nn_state_pred = None, None
        nn_output_mem, nn_state_mem = None, None

        if self.has_recurrent_nn:
            nn_output_pred, nn_state_pred, nn_output_mem, nn_state_mem = self.network_op(complete_input, state_mem, state_pred)
        else:
            nn_output_pred, nn_output_mem = self.network_op(complete_input)

        pre_output = tf.matmul(nn_output_pred, self.nn_output_weights)
        interface = tf.matmul(nn_output_mem, self.interface_weights)
        parsed_interface = self.parse_interface_vector(interface)

        if self.has_recurrent_nn:
            return pre_output, parsed_interface, nn_state_mem, nn_state_pred
        else:
            return pre_output, parsed_interface

    def network_vars(self):
        self.layer = 3
        self.node = 512

        # self.initializer = tf.random_normal_initializer(stddev=0.1)
        self.initializer = tf.random_normal_initializer(stddev=2 / (self.input_size + self.output_size))
        # self.initializer = tf.orthogonal_initializer()``

        self.lstm_cell_mem = []
        self.lstm_cell_pred = []

        for _ in range(self.layer):
            lstm_cell = WeightNormLSTMCell(
                self.node,
                # state_is_tuple=True,
                activation=selu,
            )
            lstm_cell_p = WeightNormLSTMCell(
                self.node,
                # state_is_tuple=True,
                activation=selu,
            )
            # lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=0.7)
            if _ > 0:
                lstm_cell = tf.nn.rnn_cell.ResidualWrapper(lstm_cell)
                lstm_cell_p = tf.nn.rnn_cell.ResidualWrapper(lstm_cell_p)

            self.lstm_cell_mem.append(lstm_cell)
            self.lstm_cell_pred.append(lstm_cell_p)

        if self.layer > 1:
            self.stack_lstm_mem = tf.contrib.rnn.MultiRNNCell(self.lstm_cell_mem, state_is_tuple=True)
            self.stack_lstm_pred = tf.contrib.rnn.MultiRNNCell(self.lstm_cell_mem, state_is_tuple=True)
        else:
            self.stack_lstm_mem = self.lstm_cell_mem[0]
            self.stack_lstm_pred = self.lstm_cell_mem[0]

        self.state_mem = self.stack_lstm_mem.zero_state(self.batch_size, tf.float32)
        self.state_pred = self.stack_lstm_pred.zero_state(self.batch_size, tf.float32)

    def network_op(self, X, state_mem, state_pred):
        X = tf.convert_to_tensor(X)
        print('-' * 100)
        print(X)
        print('-' * 100)
        Y_mem, new_state_mem = self.stack_lstm_mem(X, state_mem)
        Y_pred, new_state_pred = self.stack_lstm_pred(X, state_pred)

        # Y = tf.contrib.layers.layer_norm(Y)
        return Y_pred, new_state_pred, Y_mem, new_state_mem

    def get_state(self):
        return self.state_mem, self.state_pred

    def get_nn_output_size(self):
        """
        retrives the output size of the defined neural network

        Returns: int
            the output's size

        Raises: ValueError
        """

        input_vector = np.zeros([self.batch_size, self.nn_input_size], dtype=np.float32)
        output_vector = None

        if self.has_recurrent_nn:
            stat_mem, stat_pred = self.get_state()
            output_vector, _, output_vector2, __ = self.network_op(input_vector, stat_mem, stat_pred)
        else:
            output_vector = self.network_op(input_vector)

        shape = output_vector.get_shape().as_list()

        if len(shape) > 2:
            raise ValueError("Expected the neural network to output a 1D vector, but got %dD" % (len(shape) - 1))
        else:
            return shape[1]

    # def final_output(self, pre_output, new_read_vectors):
    #     """
    #     We are not using pre_output here, so it will always be None.
    #     """
    #     flat_read_vectors = tf.reshape(new_read_vectors, (-1, self.word_size * self.read_heads))
    #     final_output = tf.concat([pre_output, tf.matmul(flat_read_vectors, self.mem_output_weights)], 1)
    #
    #     final_output = tf.contrib.layers.fully_connected(
    #         final_output,
    #         self.output_size,
    #         activation_fn=tf.nn.relu
    #     )
    #     return final_output


class AutoController(L2RecurrentController):
    """
    AutoController is part DNC controller part autoencoder.
    This controller try to create useful write_vector by
    using part of controller network to reconstruct input_data from write_vector.
    """
    def decoder_vars(self, input_data):

        self.decode_layer = [tf.contrib.layers.fully_connected(input_data, self.word_size)]
        output = self.decode_layer[-1]

        node_inc = (self.input_size - self.word_size) / (self.layer - 1)
        for i in range(1, self.layer):
            self.decode_layer.append(
                tf.contrib.layers.fully_connected(
                    self.decode_layer[-1],
                    self.word_size + math.ceil(i * node_inc)
                )
            )
            output = self.decode_layer[-1]

        return output

    def process_input(self, X, last_read_vectors, state=None):
        """
        X: Tensor (batch_size, input_size)
            the input data batch
        last_read_vectors: (batch_size, word_size, read_heads)
            the last batch of read vectors from memory
        state: Tuple
            state vectors if the network is recurrent

        Returns: Tuple
            pre-output: Tensor (batch_size, output_size)
            parsed_interface_vector: dict
        """

        flat_read_vectors = tf.reshape(last_read_vectors, (-1, self.word_size * self.read_heads))
        complete_input = tf.concat([X, flat_read_vectors], 1)
        nn_output, nn_state = None, None

        if self.has_recurrent_nn:
            nn_output, nn_state = self.network_op(complete_input, state)
        else:
            nn_output = self.network_op(complete_input)

        if self.DEBUG:
            self.nn_output_weights = tf.Print(self.nn_output_weights, [self.nn_output_weights], 'self.nn_output_weights=')
            self.nn_output_weights = tf.Print(self.nn_output_weights, [tf.reduce_mean(self.nn_output_weights)], 'self.nn_output_weights mean=')
            self.nn_output_weights = tf.Print(self.nn_output_weights, [tf.shape(self.nn_output_weights)], 'self.nn_output_weights shape=')
            nn_output = tf.Print(nn_output, [nn_output], 'nn_output=')
            nn_output = tf.Print(nn_output, [tf.reduce_mean(nn_output)], 'nn_output mean=')
            nn_output = tf.Print(nn_output, [tf.shape(nn_output)], 'nn_output shape=')

        pre_output = tf.matmul(nn_output, self.nn_output_weights)
        interface = tf.matmul(nn_output, self.interface_weights)
        parsed_interface = self.parse_interface_vector(interface)

        decoder_out = self.decoder_vars(parsed_interface['write_vector'])

        # if getattr(self, 'interal_loss', None) is None:
        #     self.interal_loss = tf.losses.mean_squared_error(decoder_out, X)
        # else:
        #     self.interal_loss += tf.losses.mean_squared_error(decoder_out, X)

        if self.has_recurrent_nn:
            return pre_output, parsed_interface, nn_state, decoder_out
        else:
            return pre_output, parsed_interface, decoder_out

    def final_output(self, pre_output, new_read_vectors):
        """
        returns the final output by taking rececnt memory changes into account

        Parameters:
        ----------
        pre_output: Tensor (batch_size, output_size)
            the ouput vector from the input processing step
        new_read_vectors: Tensor (batch_size, words_size, read_heads)
            the newly read vectors from the updated memory

        Returns: Tensor (batch_size, output_size)
        """

        # pre_output = tf.Print(pre_output, [pre_output], 'pre_output=')
        # pre_output = tf.Print(pre_output, [tf.reduce_mean(pre_output)], 'pre_output mean=')

        # self.mem_output_weights = tf.Print(self.mem_output_weights, [self.mem_output_weights], 'self.mem_output_weights=')
        # self.mem_output_weights = tf.Print(self.mem_output_weights, [tf.reduce_mean(self.mem_output_weights)], 'self.mem_output_weights mean=')

        flat_read_vectors = tf.reshape(new_read_vectors, (-1, self.word_size * self.read_heads))
        # flat_read_vectors = tf.Print(flat_read_vectors, [flat_read_vectors], 'flat_read_vectors=')
        # flat_read_vectors = tf.Print(flat_read_vectors, [tf.reduce_mean(flat_read_vectors)], 'flat_read_vectors mean=')

        final_output = pre_output + tf.matmul(flat_read_vectors, self.mem_output_weights)

        # final_output = tf.Print(final_output, [final_output], 'final_output=')
        # final_output = tf.Print(final_output, [tf.reduce_mean(final_output)], 'final_output mean=')
        return final_output
