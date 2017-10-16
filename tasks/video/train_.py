import warnings
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import pickle
import getopt
import time
import sys
import os
import csv
import spacy
import re
import math
import random

from dnc.dnc import *
from recurrent_controller import *
from post_controller import *
from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image
from termcolor import colored
from train_until import *
from dataset_generator import VideoCaptionDataset

anno_file = './dataset/MSR_en.csv'
dict_file = './dataset/MSR_BBC_EN_dict.csv'
# dict_file = './dataset/MSR_en_dict.csv'
w2v_dict_file = './dataset/MSR_enW2V_dict.csv'
video_dir = './dataset/YouTubeClips/'
word2v_emb_file = './dataset/MSR_enW2V.npy'


if __name__ == '__main__':

    dirname = os.path.dirname(__file__)
    ckpts_dir = os.path.join(dirname, 'checkpoints')
    data_dir = os.path.join(dirname, 'data', 'en-10k')
    tb_logs_dir = os.path.join(dirname, 'logs')

    feat_files = [re.match('features_(\d+)_(\d+)\.npy', f) for f in os.listdir(path='./dataset/')]
    feat_files_tup = []
    for f in feat_files:
        if f is not None:
            feat_files_tup.append((f.string, int(f.group(1)), int(f.group(2))))  # (file_name, start_id, end_id)
    feat_files_tup.sort(key=lambda x: x[1])  # sort by start data id.

    llprint("Loading Data ... ")

    # w2v_emb = np.load(word2v_emb_file) * 10  # [word_num, vector_len]
    w2v_emb = None
    # Hamming = 64
    Hamming = None
    mul_onehot = (256, 2)
    # mul_onehot_map = mul_onehot_remap(config=mul_onehot_map)

    if w2v_emb is not None:
        data, lexicon_dict = load(anno_file, w2v_dict_file)
        word_space_size = output_size = w2v_emb.shape[1]
    elif Hamming is not None:
        data, lexicon_dict = load(anno_file, dict_file)
        word_space_size = output_size = Hamming
    elif mul_onehot is not None:
        data, lexicon_dict = load(anno_file, dict_file)
        word_space_size = output_size = mul_onehot[0] * mul_onehot[1]
    else:
        data, lexicon_dict = load(anno_file, dict_file)
        word_space_size = output_size = len(lexicon_dict)

    bbc_croups = load_json('dataset/bbc_croups.json')
    sequence_max_length = 500

    llprint("Done!\n")

    batch_size = 1
    input_size = 2048
    words_count = 512
    word_size = 512
    read_heads = 4

    learning_rate = 1e-4
    momentum = 0.8

    from_checkpoint = None
    iterations = len(data)
    data_size = 30000
    start_step = 0

    last_sum = 1
    last_log = 1
    mis_data_offset = 0

    # execution mode
    single_repeat = False
    feedback = True
    DEBUG = False
    TEST = False
    show_sentence = False

    options, _ = getopt.getopt(sys.argv[1:], '', ['checkpoint=', 'iterations=', 'start=', 'sig_vertifi=', 'debug=', 'test=', 'show_sentence='])

    """
    ༼ つ ◕_◕ ༽つ   ❤ ☀ ☆ ☂ ☻ ♞ ☯
    """

    for opt in options:
        if opt[0] == '--checkpoint':
            from_checkpoint = opt[1]
        elif opt[0] == '--iterations':
            iterations = int(opt[1])
        elif opt[0] == '--start':
            last_sum = last_log = start_step = int(opt[1])
        elif opt[0] == '--sig_vertifi':
            lowerc = opt[1].lower()
            single_repeat = lowerc == 't' or lowerc == 'true' or lowerc == '1'
        elif opt[0] == '--debug':
            lowerc = opt[1].lower()
            DEBUG = lowerc == 't' or lowerc == 'true' or lowerc == '1'
        elif opt[0] == '--test':
            lowerc = opt[1].lower()
            TEST = lowerc == 't' or lowerc == 'true' or lowerc == '1'
        elif opt[0] == '--show_sentence':
            lowerc = opt[1].lower()
            show_sentence = lowerc == 't' or lowerc == 'true' or lowerc == '1'

    graph = tf.Graph()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with graph.as_default():
        with tf.Session(graph=graph, config=config) as session:

            llprint("Building Computational Graph ... ")

            llprint("Done!")
            llprint("Building DNC ... ")

            # ncomputer = DNC(
            #     L2RecurrentController,
            #     input_size,
            #     output_size,
            #     sequence_max_length,
            #     words_count,
            #     word_size,
            #     read_heads,
            #     batch_size,
            #     output_feedback=feedback
            # )

            ncomputer = DNCDuo(
                MemRNNController,
                input_size,
                output_size,
                sequence_max_length,
                words_count,
                word_size,
                read_heads,
                batch_size,
                testing=TEST,
                output_feedback=feedback
            )

            output, memory_view = ncomputer.get_outputs()
            softmax_output = tf.nn.softmax(output)
            memory_states = ncomputer.get_memoory_states()

            # optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=momentum)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate)

            loss_weights = tf.placeholder(tf.float32, [batch_size, None])
            target_range = tf.placeholder_with_default([[1, 1] for _ in range(batch_size)], [batch_size, 2])  # start index, target len

            zero = tf.constant(0, dtype=tf.float32)
            # output tensors will containing all output from both input steps and output steps.

            """
            Loss functions
            """

            loss_decode = None
            if w2v_emb is not None:
                loss = tf.losses.absolute_difference(output * tf.expand_dims(loss_weights, axis=2), ncomputer.target_output)
                loss /= tf.reduce_sum(loss_weights)
                # flat_read_vectors = tf.reshape(new_read_vectors, (-1, word_size * read_heads))
            elif Hamming is not None:
                # TODO: code binary CE loss
                loss = tf.contrib.keras.backend.binary_crossentropy(
                    output,
                    ncomputer.target_output,
                    from_logits=True
                )
                loss = tf.reduce_sum(loss)
                total_size = tf.reduce_sum(loss_weights)
                total_size += 1e-12  # to avoid division by 0 for all-0 weights
                loss /= total_size
            elif mul_onehot is not None:
                target_output_mul_id = tf.placeholder(tf.int32, [batch_size, None, mul_onehot[1]], name='targets_mul_id')
                softmax_slice = []
                softout_slice = []

                for i in range(mul_onehot[1]):
                    st = i * mul_onehot[0]
                    ed = (i + 1) * mul_onehot[0]
                    seq_loss = tf.contrib.seq2seq.sequence_loss(output[:, :, st:ed], target_output_mul_id[:, :, i], loss_weights)
                    softmax_slice.append(seq_loss)
                    softout_slice.append(tf.nn.softmax(output[:, :, st:ed]))

                loss = tf.reduce_sum(tf.stack(softmax_slice))
                softmax_output = tf.concat(softout_slice, 2)
            else:  # using one-hot embedding
                loss = tf.contrib.seq2seq.sequence_loss(output, ncomputer.target_output_id, loss_weights)

                """
                loss for DNCAuto's write_vecotr autoencoder.
                """
                if type(ncomputer) is DNCAuto:
                    loss_decode = tf.losses.absolute_difference(
                        ncomputer.input_data,
                        ncomputer.get_decoder_output()
                    )
                    loss += loss_decode

            summeries = []

            gradients = optimizer.compute_gradients(loss)

            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    # TODO: add grad noise base on "saddle point condition"
                    noise = tf.random_normal(tf.shape(var), stddev=1e-3)
                    gradients[i] = (tf.clip_by_value(grad, -5, 5) + noise, var)

            for (grad, var) in gradients:
                if grad is not None:
                    summeries.append(tf.summary.histogram(var.name + '/grad', grad))

            trainable_var = tf.trainable_variables()
            for v in trainable_var:
                summeries.append(tf.summary.histogram(v.name + '/values', v))

            apply_gradients = optimizer.apply_gradients(gradients)

            summeries.append(tf.summary.scalar("Loss", loss))

            summerize_op = tf.summary.merge(summeries)
            no_summerize = tf.no_op()
            summerizer = tf.summary.FileWriter(tb_logs_dir, graph)

            llprint("Done!")

            llprint("Initializing Variables ... ")
            session.run(tf.global_variables_initializer())
            llprint("Done!")

            if from_checkpoint is not None:
                llprint("Restoring Checkpoint %s ... " % (from_checkpoint))
                ncomputer.restore(session, ckpts_dir, from_checkpoint)
                llprint("Done!")

            # session = tf_debug.LocalCLIDebugWrapperSession(session)
            # session.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

            last_100_losses = []
            last_avg_min_max = [0, 0, 0]

            start = 0 if start_step == 0 else start_step + 1
            end = 4000000
            # end = start_step + iterations + 1 if start_step + iterations + 1 < len(data) else len(data)
            reuse_data_param = 1

            start_time_100 = time.time()
            end_time_100 = None
            avg_100_time = 0.
            avg_counter = 0

            dataset = VideoCaptionDataset('./dataset', data, lexicon_dict, dataset_size=data_size, show_log=True, batch_size=batch_size, croups=bbc_croups)

            for i, batch in zip(range(start, end), dataset.get_batchs(start_iter=start, feedback=feedback, word_emb=w2v_emb, mul_onehot=mul_onehot, norm=True)):
                try:

                    input_data = batch['input_data']
                    target_outputs = batch['target_outputs']
                    seq_len = batch['seq_len']['seq_len']
                    mask = batch['mask']
                    target_step = None

                    summerize = (i - last_sum >= 100)
                    runtime_statistics = (i % 1000 == 0) and False
                    # take_checkpoint = (i != 0) and (i % 200 == 0)

                    print('Feed features into DNC.')

                    # reapeating same input for 'seq_reapte' times.
                    # for n in range(seq_reapte):
                    first_loss = None
                    n = 0

                    if runtime_statistics:
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()

                        feed = {
                            ncomputer.input_data: input_data,
                            ncomputer.target_output: target_outputs,
                            ncomputer.target_output_id: onehot_vec2id(target_outputs),
                            ncomputer.sequence_length: seq_len,
                            loss_weights: mask.reshape([batch_size, -1]),
                            # target_range: target_step
                        }

                        if mul_onehot2ids is not None:
                            feed[target_output_mul_id] = mul_onehot2ids(target_outputs)

                        loss_de_value, loss_value, out_value, _, summary = session.run([
                            loss_decode if loss_decode is not None else tf.no_op(),
                            loss,
                            softmax_output,
                            apply_gradients,
                            summerize_op if summerize else no_summerize,
                        ], feed_dict=feed,
                            options=run_options,
                            run_metadata=run_metadata
                        )

                        summerizer.add_run_metadata(run_metadata, 'step%d' % i)

                    elif DEBUG:
                        feed = {
                            ncomputer.input_data: input_data,
                            ncomputer.target_output: target_outputs,
                            ncomputer.target_output_id: onehot_vec2id(target_outputs),
                            ncomputer.sequence_length: seq_len,
                            loss_weights: mask.reshape([batch_size, -1]),
                            # target_range: target_step
                        }

                        if mul_onehot2ids is not None:
                            feed[target_output_mul_id] = mul_onehot2ids(target_outputs)

                        loss_de_value, loss_value, out_value, raw_output, grads = session.run([
                            loss_decode if loss_decode is not None else tf.no_op(),
                            loss,
                            softmax_output,
                            output,
                            gradients,
                        ], feed_dict=feed)

                        debug_var = {
                            "loss": loss_value,
                            "softmax_out": out_value,
                            "raw_out": raw_output,
                            "mask": mask.reshape([batch_size, -1]),
                            "target": target_outputs,
                            "grad": {var[1].name: v for var, v in zip(gradients, grads)},
                        }

                        Target_sent = decode_output(lexicon_dict, target_outputs[0], word2v_emb=w2v_emb, hamming=Hamming, mul_onehot=mul_onehot)
                        print(colored('Target: ', color='cyan'), Target_sent[0])

                        DNC_sent = decode_output(lexicon_dict, out_value[0], target=Target_sent[0], word2v_emb=w2v_emb, mul_onehot=mul_onehot)
                        for out in DNC_sent:
                            print(colored('DCN: ', color='green'), out)

                        summary = None
                        np.save("debug_%s.npy" % from_checkpoint, debug_var)
                        sys.exit(0)
                    elif TEST:
                        out_value, raw_output, mem_tuple, mem_matrix = session.run([
                            softmax_output,
                            output,
                            memory_view,
                            memory_states,
                        ], feed_dict={
                            ncomputer.input_data: input_data,
                            ncomputer.target_output: target_outputs,
                            ncomputer.target_output_id: onehot_vec2id(target_outputs),
                            ncomputer.sequence_length: seq_len,
                            loss_weights: mask.reshape([batch_size, -1]),
                            # target_range: target_step
                        })

                        Target_sent = decode_output(lexicon_dict, target_outputs[0], word2v_emb=w2v_emb, hamming=Hamming, mul_onehot=mul_onehot)
                        print(colored('Target: ', color='cyan'), Target_sent[0])

                        DNC_sent = decode_output(lexicon_dict, out_value[0], target=Target_sent[0], word2v_emb=w2v_emb, mul_onehot=mul_onehot)
                        for out in DNC_sent:
                            print(colored('DCN: ', color='green'), out)

                        summary = None

                        np.save(os.path.join('./Visualize', get_video_name(sample) + '_memView_%s.npy' % from_checkpoint), mem_tuple)
                        np.save(os.path.join('./Visualize', get_video_name(sample) + '_memMatrix_%s.npy' % from_checkpoint), mem_matrix)
                        np.save(os.path.join('./Visualize', get_video_name(sample) + '_outputMatrix_%s.npy' % from_checkpoint), out_value)
                        # if i >= start + 10:
                        sys.exit(0)

                    else:
                        """
                        Training
                        """

                        feed = {
                            ncomputer.input_data: input_data,
                            ncomputer.target_output: target_outputs,
                            ncomputer.target_output_id: onehot_vec2id(target_outputs),
                            ncomputer.sequence_length: seq_len,
                            loss_weights: mask.reshape([batch_size, -1]),
                            # target_range: target_step
                        }

                        if mul_onehot2ids is not None:
                            feed[target_output_mul_id] = mul_onehot2ids(target_outputs)

                        loss_value, out_value, _, summary = session.run([
                            # loss_decode if loss_decode is not None else tf.no_op(),
                            loss,
                            softmax_output,
                            apply_gradients,
                            summerize_op if summerize else no_summerize,
                        ], feed_dict=feed)

                        print(colored('[%d]Loss: ' % n, color='green'), loss_value)
                        print('-' * 100)

                        last_100_losses.append(loss_value)

                        if summary is not None:
                            summerizer.add_summary(summary, i)

                        if i - last_sum >= 100:
                            last_sum = i
                            llprint("   Avg. Cross-Entropy: %.7f" % (np.mean(last_100_losses)))
                            llprint("   Max. %.7f  Min. %.7f" % (max(last_100_losses), min(last_100_losses)))
                            last_avg_min_max = [np.mean(last_100_losses), min(last_100_losses), max(last_100_losses)]

                            end_time_100 = time.time()
                            elapsed_time = (end_time_100 - start_time_100) / 60
                            avg_counter += 1
                            avg_100_time += (1. / avg_counter) * (elapsed_time - avg_100_time)
                            estimated_time = (avg_100_time * ((end - i) / 100.)) / 60.

                            print("Avg. 100 iterations time: %.2f minutes" % (avg_100_time))
                            print("Approx. time to completion: %.2f hours" % (estimated_time))

                            try:
                                os.system("python3 PyGoogleSheet/pyGooSheet.py --step %d --value %.7f" % (i, np.mean(last_100_losses)))
                            except:
                                print(colored('Error: ', color='red'), 'fail to update google sheet!')

                            start_time_100 = time.time()
                            last_100_losses = []

                        if i - last_log >= 1000:
                            last_log = i
                            llprint("Saving Checkpoint ... "),
                            ncomputer.save(session, ckpts_dir, 'step-%d' % (i))
                            llprint("Done!")

                except KeyboardInterrupt:

                    llprint("Saving Checkpoint ... "),
                    ncomputer.save(session, ckpts_dir, 'step-%d' % (i))
                    llprint("Done!")
                    sys.exit(0)
