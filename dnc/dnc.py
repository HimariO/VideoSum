import tensorflow as tf
from tensorflow.contrib.rnn import LSTMStateTuple
from .memory import *
from .utility import *
import os


class DNC:

    def __init__(self, controller_class, input_size, output_size, max_sequence_length,
                 memory_words_num=256, memory_word_size=64, memory_read_heads=4, batch_size=1, testing=False, output_feedback=False):
        """
        constructs a complete DNC architecture as described in the DNC paper
        http://www.nature.com/nature/journal/vaop/ncurrent/full/nature20101.html

        Parameters:
        -----------
        controller_class: BaseController
            a concrete implementation of the BaseController class
        input_size: int
            the size of the input vector
        output_size: int
            the size of the output vector
        max_sequence_length: int
            the maximum length of an input sequence
        memory_words_num: int
            the number of words that can be stored in memory
        memory_word_size: int
            the size of an individual word in memory
        memory_read_heads: int
            the number of read heads in the memory
        batch_size: int
            the size of the data batch
        """
        self.testing = testing
        self.feedback = output_feedback
        self.output_t = tf.zeros([batch_size, output_size])

        self.input_size = input_size + output_size if output_feedback else input_size
        self.output_size = output_size
        self.max_sequence_length = max_sequence_length
        self.words_num = memory_words_num
        self.word_size = memory_word_size
        self.read_heads = memory_read_heads
        self.batch_size = batch_size

        self.memory = SharpMemory(self.words_num, self.word_size, self.read_heads, self.batch_size)
        self.packed_memory_matrixs = {}
        self.controller = controller_class(self.input_size, self.output_size, self.read_heads, self.word_size, self.batch_size)

        # input data placeholders
        self.input_data = tf.placeholder(tf.float32, [batch_size, None, self.input_size], name='input')
        self.target_output = tf.placeholder(tf.float32, [batch_size, None, self.output_size], name='targets')
        self.target_output_id = tf.placeholder(tf.int32, [batch_size, None], name='targets_id')
        # self.target_output_mul_id = tf.placeholder(tf.int32, [batch_size, None, ], name='targets_id')
        self.sequence_length = tf.placeholder(tf.int32, name='sequence_length')

        self.penalty_term = None
        self.build_graph()

    def _step_op(self, step, memory_state, controller_state=None):
        """
        performs a step operation on the input step data

        Parameters:
        ----------
        step: Tensor (batch_size, input_size)
        memory_state: Tuple
            a tuple of current memory parameters
        controller_state: Tuple
            the state of the controller if it's recurrent

        Returns: Tuple
            output: Tensor (batch_size, output_size)
            memory_view: dict
        """

        last_read_vectors = memory_state[6]
        pre_output, interface, nn_state = None, None, None

        if self.controller.has_recurrent_nn:
            pre_output, interface, nn_state = self.controller.process_input(step, last_read_vectors, controller_state)
        else:
            pre_output, interface = self.controller.process_input(step, last_read_vectors)

        usage_vector, write_weighting, memory_matrix, link_matrix, precedence_vector = self.memory.write(
            memory_state[0], memory_state[1], memory_state[5],
            memory_state[4], memory_state[2], memory_state[3],
            interface['write_key'],
            interface['write_strength'],
            interface['free_gates'],
            interface['allocation_gate'],
            interface['write_gate'],
            interface['write_vector'],
            interface['erase_vector']
        )

        if isinstance(self.memory, SharpMemory) or isinstance(self.memory, KMemory):
            read_weightings, read_vectors = self.memory.read(
                memory_matrix,
                memory_state[5],
                interface['read_keys'],
                interface['read_strengths'],
                link_matrix,
                interface['read_modes'],
                memory_state[1],
            )
        else:
            read_weightings, read_vectors = self.memory.read(
                memory_matrix,
                memory_state[5],
                interface['read_keys'],
                interface['read_strengths'],
                link_matrix,
                interface['read_modes'],
            )

        return [

            # report new memory state to be updated outside the condition branch
            memory_matrix,
            usage_vector,
            precedence_vector,
            link_matrix,
            write_weighting,
            read_weightings,
            read_vectors,

            self.controller.final_output(pre_output, read_vectors),
            interface['free_gates'],
            interface['allocation_gate'],
            interface['write_gate'],

            # report new state of RNN if exists
            nn_state if nn_state is not None else tf.zeros(1),
        ]


    def _loop_body(self, time, memory_state, outputs, free_gates, allocation_gates, write_gates,
                   read_weightings, write_weightings, usage_vectors, controller_state, *memory_state_record):
        """
        the body of the DNC sequence processing loop

        Parameters:
        ----------
        time: Tensor
        outputs: TensorArray
        memory_state: Tuple
        free_gates: TensorArray
        allocation_gates: TensorArray
        write_gates: TensorArray
        read_weightings: TensorArray,
        write_weightings: TensorArray,
        usage_vectors: TensorArray,
        controller_state: Tuple

        Returns: Tuple containing all updated arguments
        """

        step_input = self.unpacked_input_data.read(time)

        if self.feedback:
            # redirect output if DNC is at test time(trainging time will use target as feedback so no need to change TF graph)
            if time == 0 and self.testing:
                #  get input data with out feedback part.
                step_input = tf.slice(step_input, [0, 0], [self.batch_size, self.input_size - self.output_size])
                step_input = tf.concat([step_input, tf.zeros([self.batch_size, self.output_size])], 1)
            elif self.testing:
                step_input = tf.slice(step_input, [0, 0], [self.batch_size, self.input_size - self.output_size])
                step_input = tf.concat([step_input, self.output_t], 1)

        output_list = self._step_op(step_input, memory_state, controller_state)

        # update memory parameters

        new_controller_state = tf.zeros(1)
        new_memory_state = tuple(output_list[0:7])

        new_controller_state = output_list[11]

        outputs = outputs.write(time, output_list[7])
        self.output_t = output_list[7]

        # collecting memory view for the current step
        free_gates = free_gates.write(time, output_list[8])
        allocation_gates = allocation_gates.write(time, output_list[9])
        write_gates = write_gates.write(time, output_list[10])
        read_weightings = read_weightings.write(time, output_list[5])
        write_weightings = write_weightings.write(time, output_list[4])
        usage_vectors = usage_vectors.write(time, output_list[1])

        memory_state_list = list(memory_state_record)
        if self.testing:
            memory_state_list[0] = memory_state_record[0].write(time, new_memory_state[0])
            memory_state_list[1] = memory_state_record[1].write(time, new_memory_state[1])
            memory_state_list[2] = memory_state_record[2].write(time, new_memory_state[2])
            memory_state_list[3] = memory_state_record[3].write(time, new_memory_state[3])
            memory_state_list[4] = memory_state_record[4].write(time, new_memory_state[4])
            memory_state_list[5] = memory_state_record[5].write(time, new_memory_state[5])
            memory_state_list[6] = memory_state_record[6].write(time, new_memory_state[6])

        return (
            time + 1, new_memory_state, outputs,
            free_gates, allocation_gates, write_gates,
            read_weightings, write_weightings,
            usage_vectors, new_controller_state,
            *memory_state_list
        )

    def build_graph(self):
        """
        builds the computational graph that performs a step-by-step evaluation
        of the input data batches
        """

        self.unpacked_input_data = unpack_into_tensorarray(self.input_data, 1, self.sequence_length)

        outputs = tf.TensorArray(tf.float32, self.sequence_length)
        usage_vectors = tf.TensorArray(tf.float32, self.sequence_length)

        free_gates = tf.TensorArray(tf.float32, self.sequence_length)
        allocation_gates = tf.TensorArray(tf.float32, self.sequence_length)
        write_gates = tf.TensorArray(tf.float32, self.sequence_length)

        read_weightings = tf.TensorArray(tf.float32, self.sequence_length)
        write_weightings = tf.TensorArray(tf.float32, self.sequence_length)

        memory_state = self.memory.init_memory()
        memory_state_record = [
            tf.TensorArray(tf.float32, self.sequence_length),
            tf.TensorArray(tf.float32, self.sequence_length),
            tf.TensorArray(tf.float32, self.sequence_length),
            tf.TensorArray(tf.float32, self.sequence_length),
            tf.TensorArray(tf.float32, self.sequence_length),
            tf.TensorArray(tf.float32, self.sequence_length),
            tf.TensorArray(tf.float32, self.sequence_length),
        ]

        controller_state = self.controller.get_state() if self.controller.has_recurrent_nn else (tf.zeros(1), tf.zeros(1))
        # This 2 line of code will cause problem if controller have more than 1 layer.
        # if not isinstance(controller_state, LSTMStateTuple):
        #     controller_state = LSTMStateTuple(controller_state[0], controller_state[1])

        final_results = None

        time = tf.constant(0, dtype=tf.int32)

        final_results = tf.while_loop(
            cond=lambda time, *_: time < self.sequence_length,
            body=self._loop_body,
            loop_vars=(
                time, memory_state, outputs,
                free_gates, allocation_gates, write_gates,
                read_weightings, write_weightings,
                usage_vectors, controller_state, *memory_state_record,
            ),
            parallel_iterations=32,
            swap_memory=True
        )

        dependencies = []
        if self.controller.has_recurrent_nn:
            dependencies.append(self.controller.update_state(final_results[9]))  # result[9] is new_controller_state

        with tf.control_dependencies(dependencies):
            self.packed_output = pack_into_tensor(final_results[2], axis=1)

            self.packed_memory_view = {
                'free_gates': pack_into_tensor(final_results[3], axis=1),
                'allocation_gates': pack_into_tensor(final_results[4], axis=1),
                'write_gates': pack_into_tensor(final_results[5], axis=1),
                'read_weightings': pack_into_tensor(final_results[6], axis=1),
                'write_weightings': pack_into_tensor(final_results[7], axis=1),
                'usage_vectors': pack_into_tensor(final_results[8], axis=1)
            }
            if self.testing:
                self.packed_memory_matrixs = {
                    'memory_matrix': pack_into_tensor(final_results[10], axis=1),
                    'usage_vector': pack_into_tensor(final_results[11], axis=1),
                    'precedence_vector': pack_into_tensor(final_results[12], axis=1),
                    'link_matrix': pack_into_tensor(final_results[13], axis=1),
                    'write_weighting': pack_into_tensor(final_results[14], axis=1),
                    'read_weightings': pack_into_tensor(final_results[15], axis=1),
                    'read_vectors': pack_into_tensor(final_results[16], axis=1),
                }

    def get_outputs(self):
        """
        returns the graph nodes for the output and memory view

        Returns: Tuple
            outputs: Tensor (batch_size, time_steps, output_size)
            memory_view: dict
        """
        return self.packed_output, self.packed_memory_view

    def get_memoory_states(self):
        """
        returns 'memory_matrix','usage_vector','precedence_vector','link_matrix',
        'write_weighting','read_weightings','read_vectors'

        Returns: Tuple
            packed_memory_matrixs: dict
        """
        return self.packed_memory_matrixs

    def save(self, session, ckpts_dir, name):
        """
        saves the current values of the model's parameters to a checkpoint

        Parameters:
        ----------
        session: tf.Session
            the tensorflow session to save
        ckpts_dir: string
            the path to the checkpoints directories
        name: string
            the name of the checkpoint subdirectory
        """
        checkpoint_dir = os.path.join(ckpts_dir, name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        tf.train.Saver(tf.trainable_variables()).save(session, os.path.join(checkpoint_dir, 'model.ckpt'))

    def restore(self, session, ckpts_dir, name):
        """
        session: tf.Session
            the tensorflow session to restore into
        ckpts_dir: string
            the path to the checkpoints directories
        name: string
            the name of the checkpoint subdirectory
        """
        tf.train.Saver(tf.trainable_variables()).restore(session, os.path.join(ckpts_dir, name, 'model.ckpt'))


class DNCAuto(DNC):
    def _step_op(self, step, memory_state, controller_state=None):
        """
        performs a step operation on the input step data

        Parameters:
        ----------
        step: Tensor (batch_size, input_size)
        memory_state: Tuple
            a tuple of current memory parameters
        controller_state: Tuple
            the state of the controller if it's recurrent

        Returns: Tuple
            output: Tensor (batch_size, output_size)
            memory_view: dict
        """

        last_read_vectors = memory_state[6]
        pre_output, interface, nn_state = None, None, None

        if self.controller.has_recurrent_nn:
            pre_output, interface, nn_state, decoder_out = self.controller.process_input(step, last_read_vectors, controller_state)
        else:
            pre_output, interface, decoder_out = self.controller.process_input(step, last_read_vectors)

        usage_vector, write_weighting, memory_matrix, link_matrix, precedence_vector = self.memory.write(
            memory_state[0], memory_state[1], memory_state[5],
            memory_state[4], memory_state[2], memory_state[3],
            interface['write_key'],
            interface['write_strength'],
            interface['free_gates'],
            interface['allocation_gate'],
            interface['write_gate'],
            interface['write_vector'],
            interface['erase_vector']
        )

        if isinstance(self.memory, SharpMemory) or isinstance(self.memory, KMemory):
            read_weightings, read_vectors = self.memory.read(
                memory_matrix,
                memory_state[5],
                interface['read_keys'],
                interface['read_strengths'],
                link_matrix,
                interface['read_modes'],
                memory_state[1],
            )
        else:
            read_weightings, read_vectors = self.memory.read(
                memory_matrix,
                memory_state[5],
                interface['read_keys'],
                interface['read_strengths'],
                link_matrix,
                interface['read_modes'],
            )

        return [

            # report new memory state to be updated outside the condition branch
            memory_matrix,
            usage_vector,
            precedence_vector,
            link_matrix,
            write_weighting,
            read_weightings,
            read_vectors,

            self.controller.final_output(pre_output, read_vectors),
            interface['free_gates'],
            interface['allocation_gate'],
            interface['write_gate'],

            # report new state of RNN if exists
            nn_state if nn_state is not None else tf.zeros(1),
            decoder_out
        ]

    def _loop_body(self, time, memory_state, outputs, decoder_outputs, free_gates, allocation_gates, write_gates,
                   read_weightings, write_weightings, usage_vectors, controller_state, *memory_state_record):
        """
        the body of the DNC sequence processing loop

        Parameters:
        ----------
        time: Tensor
        outputs: TensorArray
        memory_state: Tuple
        free_gates: TensorArray
        allocation_gates: TensorArray
        write_gates: TensorArray
        read_weightings: TensorArray,
        write_weightings: TensorArray,
        usage_vectors: TensorArray,
        controller_state: Tuple

        Returns: Tuple containing all updated arguments
        """

        step_input = self.unpacked_input_data.read(time)

        if self.feedback:
            if time == 0 and self.testing:
                step_input = tf.slice(step_input, [0, 0], [self.batch_size, self.input_size - self.output_size])
                step_input = tf.concat([step_input, tf.zeros([self.batch_size, self.output_size])], 1)
            elif self.testing:
                step_input = tf.slice(step_input, [0, 0], [self.batch_size, self.input_size - self.output_size])
                step_input = tf.concat([step_input, self.output_t], 1)

        output_list = self._step_op(step_input, memory_state, controller_state)

        # update memory parameters

        new_controller_state = tf.zeros(1)
        new_memory_state = tuple(output_list[0:7])

        new_controller_state = output_list[11]

        outputs = outputs.write(time, output_list[7])
        self.output_t = output_list[7]
        decoder_outputs = decoder_outputs.write(time, output_list[12])

        # collecting memory view for the current step
        free_gates = free_gates.write(time, output_list[8])
        allocation_gates = allocation_gates.write(time, output_list[9])
        write_gates = write_gates.write(time, output_list[10])
        read_weightings = read_weightings.write(time, output_list[5])
        write_weightings = write_weightings.write(time, output_list[4])
        usage_vectors = usage_vectors.write(time, output_list[1])

        memory_state_list = list(memory_state_record)
        if self.testing:
            memory_state_list[0] = memory_state_record[0].write(time, new_memory_state[0])
            memory_state_list[1] = memory_state_record[1].write(time, new_memory_state[1])
            memory_state_list[2] = memory_state_record[2].write(time, new_memory_state[2])
            memory_state_list[3] = memory_state_record[3].write(time, new_memory_state[3])
            memory_state_list[4] = memory_state_record[4].write(time, new_memory_state[4])
            memory_state_list[5] = memory_state_record[5].write(time, new_memory_state[5])
            memory_state_list[6] = memory_state_record[6].write(time, new_memory_state[6])

        return (
            time + 1, new_memory_state, outputs, decoder_outputs,
            free_gates, allocation_gates, write_gates,
            read_weightings, write_weightings,
            usage_vectors, new_controller_state,
            *memory_state_list
        )

    def build_graph(self):
        """
        builds the computational graph that performs a step-by-step evaluation
        of the input data batches
        """

        self.unpacked_input_data = unpack_into_tensorarray(self.input_data, 1, self.sequence_length)

        outputs = tf.TensorArray(tf.float32, self.sequence_length)
        decoder_outputs = tf.TensorArray(tf.float32, self.sequence_length)
        usage_vectors = tf.TensorArray(tf.float32, self.sequence_length)

        free_gates = tf.TensorArray(tf.float32, self.sequence_length)
        allocation_gates = tf.TensorArray(tf.float32, self.sequence_length)
        write_gates = tf.TensorArray(tf.float32, self.sequence_length)

        read_weightings = tf.TensorArray(tf.float32, self.sequence_length)
        write_weightings = tf.TensorArray(tf.float32, self.sequence_length)

        controller_state = self.controller.get_state() if self.controller.has_recurrent_nn else (tf.zeros(1), tf.zeros(1))
        memory_state = self.memory.init_memory()
        memory_state_record = [
            tf.TensorArray(tf.float32, self.sequence_length),
            tf.TensorArray(tf.float32, self.sequence_length),
            tf.TensorArray(tf.float32, self.sequence_length),
            tf.TensorArray(tf.float32, self.sequence_length),
            tf.TensorArray(tf.float32, self.sequence_length),
            tf.TensorArray(tf.float32, self.sequence_length),
            tf.TensorArray(tf.float32, self.sequence_length),
        ]

        # This 2 line of code will cause problem if controller have more than 1 layer.
        # if not isinstance(controller_state, LSTMStateTuple):
        #     controller_state = LSTMStateTuple(controller_state[0], controller_state[1])

        final_results = None

        time = tf.constant(0, dtype=tf.int32)

        final_results = tf.while_loop(
            cond=lambda time, *_: time < self.sequence_length,
            body=self._loop_body,
            loop_vars=(
                time, memory_state, outputs, decoder_outputs,
                free_gates, allocation_gates, write_gates,
                read_weightings, write_weightings,
                usage_vectors, controller_state, *memory_state_record,
            ),
            parallel_iterations=64,
            swap_memory=True
        )

        dependencies = []
        if self.controller.has_recurrent_nn:
            dependencies.append(self.controller.update_state(final_results[9]))  # result[9] is new_controller_state

        with tf.control_dependencies(dependencies):
            self.packed_output = pack_into_tensor(final_results[2], axis=1)
            self.packed_decoder_output = pack_into_tensor(final_results[3], axis=1)

            self.packed_memory_view = {
                'free_gates': pack_into_tensor(final_results[4], axis=1),
                'allocation_gates': pack_into_tensor(final_results[5], axis=1),
                'write_gates': pack_into_tensor(final_results[6], axis=1),
                'read_weightings': pack_into_tensor(final_results[7], axis=1),
                'write_weightings': pack_into_tensor(final_results[8], axis=1),
                'usage_vectors': pack_into_tensor(final_results[9], axis=1)
            }
            if self.testing:
                self.packed_memory_matrixs = {
                    'memory_matrix': pack_into_tensor(final_results[11], axis=1),
                    'usage_vector': pack_into_tensor(final_results[12], axis=1),
                    'precedence_vector': pack_into_tensor(final_results[13], axis=1),
                    'link_matrix': pack_into_tensor(final_results[14], axis=1),
                    'write_weighting': pack_into_tensor(final_results[15], axis=1),
                    'read_weightings': pack_into_tensor(final_results[16], axis=1),
                    'read_vectors': pack_into_tensor(final_results[17], axis=1),
                }

    def get_outputs(self):
        """
        Returns: Tuple
            outputs: Tensor (batch_size, time_steps, output_size)
            memory_view: dict
        """
        return self.packed_output, self.packed_memory_view

    def get_decoder_output(self):
        return self.packed_decoder_output


class DNCPostControl(DNC):

    def __init__(self, controller_class, post_controller_class, input_size, output_size, max_sequence_length,
                 memory_words_num=256, memory_word_size=64, memory_read_heads=4, batch_size=1, testing=False):
        self.post_control = post_controller_class(
            memory_word_size * memory_read_heads + batch_size * output_size,
            output_size, batch_size
        )

        self.testing = testing
        self.feedback = output_feedback
        self.output_t = tf.zeros([batch_size, output_size])

        self.input_size = input_size + output_size if output_feedback else input_size
        self.output_size = output_size
        self.max_sequence_length = max_sequence_length
        self.words_num = memory_words_num
        self.word_size = memory_word_size
        self.read_heads = memory_read_heads
        self.batch_size = batch_size

        self.memory = Memory(self.words_num, self.word_size, self.read_heads, self.batch_size)
        self.controller = controller_class(self.input_size, self.input_size, self.read_heads, self.word_size, self.batch_size)

        # input data placeholders
        self.input_data = tf.placeholder(tf.float32, [batch_size, None, input_size], name='input')
        self.target_output = tf.placeholder(tf.float32, [batch_size, None, output_size], name='targets')
        self.target_output_id = tf.placeholder(tf.int32, [batch_size, None], name='targets_id')
        self.sequence_length = tf.placeholder(tf.int32, name='sequence_length')

        self.build_graph()

    def _step_op(self, step, memory_state, controller_state=None, post_controller_state=None):

        last_read_vectors = memory_state[6]
        pre_output, interface, nn_state, post_nn_state = None, None, None, None

        if self.controller.has_recurrent_nn:
            pre_output, interface, nn_state = self.controller.process_input(step, last_read_vectors, controller_state)
        else:
            pre_output, interface = self.controller.process_input(step, last_read_vectors)

        usage_vector, write_weighting, memory_matrix, link_matrix, precedence_vector = self.memory.write(
            memory_state[0], memory_state[1], memory_state[5],
            memory_state[4], memory_state[2], memory_state[3],
            interface['write_key'],
            interface['write_strength'],
            interface['free_gates'],
            interface['allocation_gate'],
            interface['write_gate'],
            interface['write_vector'],
            interface['erase_vector']
        )

        read_weightings, read_vectors = self.memory.read(
            memory_matrix,
            memory_state[5],
            interface['read_keys'],
            interface['read_strengths'],
            link_matrix,
            interface['read_modes'],
        )

        final_out, post_nn_state = self.post_control.network_op(
            self.controller.final_output(pre_output, read_vectors),
            post_controller_state
        )

        return [

            # report new memory state to be updated outside the condition branch
            memory_matrix,
            usage_vector,
            precedence_vector,
            link_matrix,
            write_weighting,
            read_weightings,
            read_vectors,

            final_out,
            interface['free_gates'],
            interface['allocation_gate'],
            interface['write_gate'],

            # report new state of RNN if exists
            nn_state if nn_state is not None else tf.zeros(1),
            post_nn_state if nn_state is not None else tf.zeros(1),
        ]

    def _loop_body(self, time, memory_state, outputs, free_gates, allocation_gates, write_gates, read_weightings,
                   write_weightings, usage_vectors, controller_state, post_controller_state, *memory_state_record):

        step_input = self.unpacked_input_data.read(time)

        if self.feedback:
            if time == 0 and self.testing:
                step_input = tf.slice(step_input, [0, 0], [self.batch_size, self.input_size - self.output_size])
                step_input = tf.concat([step_input, tf.zeros([self.batch_size, self.output_size])], 1)
            elif self.testing:
                step_input = tf.slice(step_input, [0, 0], [self.batch_size, self.input_size - self.output_size])
                step_input = tf.concat([step_input, self.output_t], 1)

        output_list = self._step_op(step_input, memory_state, controller_state, post_controller_state)

        # update memory parameters

        new_controller_state = tf.zeros(1)
        new_post_controller_state = tf.zeros(1)
        new_memory_state = tuple(output_list[0:7])

        new_controller_state = output_list[11]
        new_post_controller_state = output_list[12]

        outputs = outputs.write(time, output_list[7])
        self.output_t = output_list[7]

        # collecting memory view for the current step
        free_gates = free_gates.write(time, output_list[8])
        allocation_gates = allocation_gates.write(time, output_list[9])
        write_gates = write_gates.write(time, output_list[10])
        read_weightings = read_weightings.write(time, output_list[5])
        write_weightings = write_weightings.write(time, output_list[4])
        usage_vectors = usage_vectors.write(time, output_list[1])

        memory_state_list = list(memory_state_record)
        if self.testing:
            memory_state_list[0] = memory_state_record[0].write(time, new_memory_state[0])
            memory_state_list[1] = memory_state_record[1].write(time, new_memory_state[1])
            memory_state_list[2] = memory_state_record[2].write(time, new_memory_state[2])
            memory_state_list[3] = memory_state_record[3].write(time, new_memory_state[3])
            memory_state_list[4] = memory_state_record[4].write(time, new_memory_state[4])
            memory_state_list[5] = memory_state_record[5].write(time, new_memory_state[5])
            memory_state_list[6] = memory_state_record[6].write(time, new_memory_state[6])

        return (
            time + 1, new_memory_state, outputs,
            free_gates, allocation_gates, write_gates,
            read_weightings, write_weightings,
            usage_vectors, new_controller_state, new_post_controller_state,
            *memory_state_list
        )

    def build_graph(self):
        """
        builds the computational graph that performs a step-by-step evaluation
        of the input data batches
        """

        self.unpacked_input_data = unpack_into_tensorarray(self.input_data, 1, self.sequence_length)

        outputs = tf.TensorArray(tf.float32, self.sequence_length)
        usage_vectors = tf.TensorArray(tf.float32, self.sequence_length)

        free_gates = tf.TensorArray(tf.float32, self.sequence_length)
        allocation_gates = tf.TensorArray(tf.float32, self.sequence_length)
        write_gates = tf.TensorArray(tf.float32, self.sequence_length)

        read_weightings = tf.TensorArray(tf.float32, self.sequence_length)
        write_weightings = tf.TensorArray(tf.float32, self.sequence_length)

        controller_state = self.controller.get_state() if self.controller.has_recurrent_nn else (tf.zeros(1), tf.zeros(1))
        post_controller_state = self.post_control.get_state()

        memory_state = self.memory.init_memory()
        memory_state_record = [
            tf.TensorArray(tf.float32, self.sequence_length),
            tf.TensorArray(tf.float32, self.sequence_length),
            tf.TensorArray(tf.float32, self.sequence_length),
            tf.TensorArray(tf.float32, self.sequence_length),
            tf.TensorArray(tf.float32, self.sequence_length),
            tf.TensorArray(tf.float32, self.sequence_length),
            tf.TensorArray(tf.float32, self.sequence_length),
        ]

        # if not isinstance(controller_state, LSTMStateTuple):
        #     controller_state = LSTMStateTuple(controller_state[0], controller_state[1])
        # if not isinstance(post_controller_state, LSTMStateTuple):
        #     post_controller_state = LSTMStateTuple(post_controller_state[0], post_controller_state[1])

        final_results = None

        time = tf.constant(0, dtype=tf.int32)

        final_results = tf.while_loop(
            cond=lambda time, *_: time < self.sequence_length,
            body=self._loop_body,
            loop_vars=(
                time, memory_state, outputs,
                free_gates, allocation_gates, write_gates,
                read_weightings, write_weightings,
                usage_vectors, controller_state, post_controller_state,
                *memory_state_record,
            ),
            parallel_iterations=32,
            swap_memory=True
        )

        dependencies = []
        if self.controller.has_recurrent_nn:
            dependencies.append(self.controller.update_state(final_results[9]))  # result[9] is new_controller_state
            dependencies.append(self.post_control.update_state(final_results[10]))  # result[9] is new_controller_state

        with tf.control_dependencies(dependencies):
            self.packed_output = pack_into_tensor(final_results[2], axis=1)

            self.packed_memory_view = {
                'free_gates': pack_into_tensor(final_results[3], axis=1),
                'allocation_gates': pack_into_tensor(final_results[4], axis=1),
                'write_gates': pack_into_tensor(final_results[5], axis=1),
                'read_weightings': pack_into_tensor(final_results[6], axis=1),
                'write_weightings': pack_into_tensor(final_results[7], axis=1),
                'usage_vectors': pack_into_tensor(final_results[8], axis=1)
            }
            if self.testing:
                self.packed_memory_matrixs = {
                    'memory_matrix': pack_into_tensor(final_results[11], axis=1),
                    'usage_vector': pack_into_tensor(final_results[12], axis=1),
                    'precedence_vector': pack_into_tensor(final_results[13], axis=1),
                    'link_matrix': pack_into_tensor(final_results[14], axis=1),
                    'write_weighting': pack_into_tensor(final_results[15], axis=1),
                    'read_weightings': pack_into_tensor(final_results[16], axis=1),
                    'read_vectors': pack_into_tensor(final_results[17], axis=1),
                }


class DNCDirectPostControl(DNCPostControl):

    def __init__(self, controller_class, post_controller_class, input_size, output_size, max_sequence_length,
                 memory_words_num=256, memory_word_size=64, memory_read_heads=4, batch_size=1, testing=False):
        self.post_control = post_controller_class(
            input_size + batch_size * input_size,
            output_size, cell_num=512
        )

        self.testing = testing

        self.input_size = input_size
        self.output_size = output_size
        self.max_sequence_length = max_sequence_length
        self.words_num = memory_words_num
        self.word_size = memory_word_size
        self.read_heads = memory_read_heads
        self.batch_size = batch_size

        self.memory = Memory(self.words_num, self.word_size, self.read_heads, self.batch_size)
        self.controller = controller_class(self.input_size, self.post_control.cell_num, self.read_heads, self.word_size, self.batch_size)

        # input data placeholders
        self.input_data = tf.placeholder(tf.float32, [batch_size, None, input_size], name='input')
        self.target_output = tf.placeholder(tf.float32, [batch_size, None, output_size], name='targets')
        self.target_output_id = tf.placeholder(tf.int32, [batch_size, None], name='targets_id')
        self.sequence_length = tf.placeholder(tf.int32, name='sequence_length')

        self.build_graph()

    def _step_op(self, step_input, memory_state, controller_state=None, post_controller_state=None):

        last_read_vectors = memory_state[6]
        pre_output, interface, nn_state, post_nn_state = None, None, None, None

        if self.controller.has_recurrent_nn:
            pre_output, interface, nn_state = self.controller.process_input(step_input, last_read_vectors, controller_state)
        else:
            pre_output, interface = self.controller.process_input(step_input, last_read_vectors)

        usage_vector, write_weighting, memory_matrix, link_matrix, precedence_vector = self.memory.write(
            memory_state[0], memory_state[1], memory_state[5],
            memory_state[4], memory_state[2], memory_state[3],
            interface['write_key'],
            interface['write_strength'],
            interface['free_gates'],
            interface['allocation_gate'],
            interface['write_gate'],
            interface['write_vector'],
            interface['erase_vector']
        )

        read_weightings, read_vectors = self.memory.read(
            memory_matrix,
            memory_state[5],
            interface['read_keys'],
            interface['read_strengths'],
            link_matrix,
            interface['read_modes'],
        )

        final_out, post_nn_state = self.post_control.network_op(
            self.controller.final_output(pre_output, read_vectors),
            step_input,
            post_controller_state
        )

        return [

            # report new memory state to be updated outside the condition branch
            memory_matrix,
            usage_vector,
            precedence_vector,
            link_matrix,
            write_weighting,
            read_weightings,
            read_vectors,

            final_out,
            interface['free_gates'],
            interface['allocation_gate'],
            interface['write_gate'],

            # report new state of RNN if exists
            nn_state if nn_state is not None else tf.zeros(1),
            post_nn_state if nn_state is not None else tf.zeros(1),
        ]


class DNCDuo(DNCPostControl):

    def __init__(self, controller_class, input_size, output_size, max_sequence_length,
                 memory_words_num=256, memory_word_size=64, memory_read_heads=4, batch_size=1, testing=False, output_feedback=False):

        self.testing = testing
        self.feedback = output_feedback
        self.output_t = tf.zeros([batch_size, output_size])

        self.input_size = input_size + output_size if output_feedback else input_size
        self.output_size = output_size
        self.max_sequence_length = max_sequence_length
        self.words_num = memory_words_num
        self.word_size = memory_word_size
        self.read_heads = memory_read_heads
        self.batch_size = batch_size

        self.memory = KMemory(self.words_num, self.word_size, self.read_heads, self.batch_size)
        self.packed_memory_matrixs = {}
        self.controller = controller_class(self.input_size, self.output_size, self.read_heads, self.word_size, self.batch_size)

        # input data placeholders
        self.input_data = tf.placeholder(tf.float32, [batch_size, None, self.input_size], name='input')
        self.target_output = tf.placeholder(tf.float32, [batch_size, None, self.output_size], name='targets')
        self.target_output_id = tf.placeholder(tf.int32, [batch_size, None], name='targets_id')
        self.sequence_length = tf.placeholder(tf.int32, name='sequence_length')

        self.build_graph()

    def _step_op(self, step, memory_state, controller_state_mem=None, controller_state_pred=None):

        last_read_vectors = memory_state[6]
        pre_output, interface, nn_state_mem, nn_state_pred = None, None, None, None

        if self.controller.has_recurrent_nn:
            pre_output, interface, nn_state_mem, nn_state_pred = self.controller.process_input(step, last_read_vectors, controller_state_mem, controller_state_pred)
        else:
            pre_output, interface = self.controller.process_input(step, last_read_vectors)

        usage_vector, write_weighting, memory_matrix, link_matrix, precedence_vector = self.memory.write(
            memory_state[0], memory_state[1], memory_state[5],
            memory_state[4], memory_state[2], memory_state[3],
            interface['write_key'],
            interface['write_strength'],
            interface['free_gates'],
            interface['allocation_gate'],
            interface['write_gate'],
            interface['write_vector'],
            interface['erase_vector']
        )

        if type(self.memory) is SharpMemory or isinstance(self.memory, KMemory):
            read_weightings, read_vectors = self.memory.read(
                memory_matrix,
                memory_state[5],
                interface['read_keys'],
                interface['read_strengths'],
                link_matrix,
                interface['read_modes'],
                memory_state[1],
            )
        else:
            read_weightings, read_vectors = self.memory.read(
                memory_matrix,
                memory_state[5],
                interface['read_keys'],
                interface['read_strengths'],
                link_matrix,
                interface['read_modes'],
            )

        final_out = self.controller.final_output(pre_output, read_vectors)

        return [

            # report new memory state to be updated outside the condition branch
            memory_matrix,
            usage_vector,
            precedence_vector,
            link_matrix,
            write_weighting,
            read_weightings,
            read_vectors,

            final_out,
            interface['free_gates'],
            interface['allocation_gate'],
            interface['write_gate'],

            # report new state of RNN if exists
            nn_state_mem if nn_state_mem is not None else tf.zeros(1),
            nn_state_pred if nn_state_pred is not None else tf.zeros(1),
        ]

    def build_graph(self):
        """
        builds the computational graph that performs a step-by-step evaluation
        of the input data batches
        """

        self.unpacked_input_data = unpack_into_tensorarray(self.input_data, 1, self.sequence_length)

        outputs = tf.TensorArray(tf.float32, self.sequence_length)
        usage_vectors = tf.TensorArray(tf.float32, self.sequence_length)

        free_gates = tf.TensorArray(tf.float32, self.sequence_length)
        allocation_gates = tf.TensorArray(tf.float32, self.sequence_length)
        write_gates = tf.TensorArray(tf.float32, self.sequence_length)

        read_weightings = tf.TensorArray(tf.float32, self.sequence_length)
        write_weightings = tf.TensorArray(tf.float32, self.sequence_length)

        controller_state_mem, controller_state_pred = self.controller.get_state() if self.controller.has_recurrent_nn else ((tf.zeros(1), tf.zeros(1)), (tf.zeros(1), tf.zeros(1)))
        # controller_state_pred = self.post_control.get_state()

        memory_state = self.memory.init_memory()
        memory_state_record = [
            tf.TensorArray(tf.float32, self.sequence_length),
            tf.TensorArray(tf.float32, self.sequence_length),
            tf.TensorArray(tf.float32, self.sequence_length),
            tf.TensorArray(tf.float32, self.sequence_length),
            tf.TensorArray(tf.float32, self.sequence_length),
            tf.TensorArray(tf.float32, self.sequence_length),
            tf.TensorArray(tf.float32, self.sequence_length),
        ]

        final_results = None

        time = tf.constant(0, dtype=tf.int32)

        final_results = tf.while_loop(
            cond=lambda time, *_: time < self.sequence_length,
            body=self._loop_body,
            loop_vars=(
                time, memory_state, outputs,
                free_gates, allocation_gates, write_gates,
                read_weightings, write_weightings,
                usage_vectors, controller_state_mem, controller_state_pred,
                *memory_state_record,
            ),
            parallel_iterations=32,
            swap_memory=True
        )

        dependencies = []
        if self.controller.has_recurrent_nn:
            dependencies.append(self.controller.update_state(final_results[9]))  # result[9] is new_controller_state
            # dependencies.append(self.post_control.update_state(final_results[10]))  # result[9] is new_controller_state

        with tf.control_dependencies(dependencies):
            self.packed_output = pack_into_tensor(final_results[2], axis=1)

            self.packed_memory_view = {
                'free_gates': pack_into_tensor(final_results[3], axis=1),
                'allocation_gates': pack_into_tensor(final_results[4], axis=1),
                'write_gates': pack_into_tensor(final_results[5], axis=1),
                'read_weightings': pack_into_tensor(final_results[6], axis=1),
                'write_weightings': pack_into_tensor(final_results[7], axis=1),
                'usage_vectors': pack_into_tensor(final_results[8], axis=1)
            }
            if self.testing:
                self.packed_memory_matrixs = {
                    'memory_matrix': pack_into_tensor(final_results[11], axis=1),
                    'usage_vector': pack_into_tensor(final_results[12], axis=1),
                    'precedence_vector': pack_into_tensor(final_results[13], axis=1),
                    'link_matrix': pack_into_tensor(final_results[14], axis=1),
                    'write_weighting': pack_into_tensor(final_results[15], axis=1),
                    'read_weightings': pack_into_tensor(final_results[16], axis=1),
                    'read_vectors': pack_into_tensor(final_results[17], axis=1),
                }
