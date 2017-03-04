import tensorflow as tf
import numpy as np


class PostController:

    def __init__(self, input_size=1024, output_size=1024, batch_size=1):
        self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(256)
        self.state = self.lstm_cell_zero_state(self.batch_size, tf.float32)

        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size

    def network_op(self, input_, last_state):
        X = tf.convert_to_tensor(input_)
        return self.lstm_cell(X, last_state)

    def get_state(self):
        return self.state

    def
