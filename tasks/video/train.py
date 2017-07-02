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

anno_file = './dataset/MSR_en.csv'
dict_file = './dataset/MSR_en_dict.csv'
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

    # w2v_emb = np.load(word2v_emb_file) * 3  # [word_num, vector_len]
    w2v_emb = None

    if w2v_emb is None:
        data, lexicon_dict = load(anno_file, dict_file)
        output_size = len(lexicon_dict)
        word_space_size = len(lexicon_dict)
    else:
        data, lexicon_dict = load(anno_file, w2v_dict_file)
        output_size = w2v_emb.shape[1]
        word_space_size = w2v_emb.shape[1]

    sequence_max_length = 500

    llprint("Done!\n")

    batch_size = 1
    input_size = 2048
    words_count = 256
    word_size = 512
    read_heads = 4

    learning_rate = 1e-4
    momentum = 0.9

    from_checkpoint = None
    iterations = len(data)
    data_size = 50000
    start_step = 0

    last_sum = 1
    last_log = 1
    mis_data_offset = 0
    single_repeat = False

    options, _ = getopt.getopt(sys.argv[1:], '', ['checkpoint=', 'iterations=', 'start=', 'sig_vertifi='])

    """
    If need to resume training from checkpoint, you must start form iteration which have same step with one of npy sample's start file.
    due to teh fact that sample in the middle of file have unknow ID(start + miss_data_num), after program restarting with 0 miss_data_num.
    ༼ つ ◕_◕ ༽つ
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

    graph = tf.Graph()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with graph.as_default():
        with tf.Session(graph=graph, config=config) as session:

            llprint("Building Computational Graph ... ")

            llprint("Done!")
            llprint("Building DNC ... ")

            # optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=momentum)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            summerizer = tf.summary.FileWriter(tb_logs_dir, graph)

            ncomputer = DNC(
                L2RecurrentController,
                input_size,
                output_size,
                sequence_max_length,
                words_count,
                word_size,
                read_heads,
                batch_size
            )
            # ncomputer = DNCDuo(
            #     MemRNNController,
            #     DirectPostController,
            #     input_size,
            #     output_size,
            #     sequence_max_length,
            #     words_count,
            #     word_size,
            #     read_heads,
            #     batch_size
            # )

            output, _ = ncomputer.get_outputs()
            softmax_output = tf.nn.softmax(output)
            # memMat = ncomputer.get_memoory_states()

            loss_weights = tf.placeholder(tf.float32, [batch_size, None, 1])
            # output tensors will containing all output from both input steps and output steps.
            if w2v_emb is None:
                # loss = tf.reduce_mean(
                #     loss_weights * tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=ncomputer.target_output)
                # )
                loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=ncomputer.target_output)) / tf.reduce_sum(loss_weights)
            else:
                loss = tf.losses.mean_squared_error(output, ncomputer.target_output, loss_weights)
                flat_read_vectors = tf.reshape(new_read_vectors, (-1, word_size * read_heads))

            summeries = []

            gradients = optimizer.compute_gradients(loss)
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    gradients[i] = (tf.clip_by_value(grad, -1, 1), var)
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
            end = 1500000
            # end = start_step + iterations + 1 if start_step + iterations + 1 < len(data) else len(data)
            reuse_data_param = 1

            start_time_100 = time.time()
            end_time_100 = None
            avg_100_time = 0.
            avg_counter = 0

            current_feat = (None, -1, -1)  # [npy, start_id, end_id]
            input_data = target_outputs = seq_len = mask = None
            included_vid = 0
            seq_reapte = 1
            seq_video_num = 1 if not single_repeat else 8

            shuffle_range = [start] + [i for i in range(start + (200 - start % 200), end, 200)]
            shuffle_range = shuffle_range + [end]
            _shuffle_range = []

            for ind in range(len(shuffle_range) - 1):
                orders = list(range(shuffle_range[ind], shuffle_range[ind + 1]))
                random.shuffle(orders)
                _shuffle_range += orders

            for i, d in zip(range(start, end), _shuffle_range):
                data_ID = int(d % data_size)

                if current_feat is None or current_feat[1] > data_ID or current_feat[2] < data_ID:
                    for f in feat_files_tup:
                        if f[1] <= data_ID and f[2] >= data_ID:
                            llprint('Loading %s...' % f[0])
                            current_feat = np.load('./dataset/' + f[0]), f[1], f[2]
                            break
                        else:
                            current_feat = (None, -1, -1)

                    if current_feat[1:] == (-1, -1) and current_feat[0] is None:  # if no data are aliviable.
                        print('Can\'t find suitable data for tarining.')
                        sys.exit(0)

                    if i >= reuse_data_param * data_size:  # update input seq len, if train on same data more than one time.
                        reuse_data_param = math.ceil(i / data_size / 2)
                        seq_video_num = 1 + ((reuse_data_param - 1) * 1) % 1

                try:
                    llprint("\rIteration %d/%d" % (i, end))

                    # sample = np.random.choice(data, 1)
                    file_index = data_ID - current_feat[1]

                    if batch_size == 1:
                        sample = data[data_ID]
                        video_feat = current_feat[0][file_index]
                    else:
                        sample = data[data_ID: data_ID + batch_size]
                        video_feat = current_feat[0][file_index: file_index + batch_size]
                        if len(video_feat) < batch_size:
                            temp = current_feat[0][:batch_size - len(video_feat)]
                            video_feat = np.concatenate([video_feat, temp])

                    try:
                        if batch_size == 1:
                            input_data_, target_outputs_, seq_len_, mask_ = prepare_mixSample(sample, lexicon_dict, video_feat, redu_sample_rate=2, word_emb=w2v_emb, concat_input=False)
                            # input_data_, target_outputs_, seq_len_, mask_ = accuTraining(input_data_, target_outputs_, mask_, seq_len_, i // data_size)
                        else:
                            input_data_, target_outputs_, seq_len_, mask_ = prepare_batch(sample, lexicon_dict, video_feat, i // data_size, redu_sample_rate=2, word_emb=w2v_emb, useMix=True, accuTrain=True)

                        input_data = np.concatenate([input_data, input_data_], axis=1) if input_data is not None else input_data_
                        target_outputs = np.concatenate([target_outputs, target_outputs_], axis=1) if target_outputs is not None else target_outputs_
                        seq_len = seq_len + seq_len_['seq_len'] if seq_len is not None else seq_len_['seq_len']
                        mask = np.concatenate([mask, mask_], axis=1) if mask is not None else mask_

                        included_vid += 1

                        if included_vid < seq_video_num:
                            continue
                        else:
                            included_vid = 0

                    except OSError:
                        print(colored('Error: ', color='red'), 'video %s doesn\'t exist.' % sample['VideoID'])
                        continue

                    except KeyError:
                        print(colored('Error: ', color='red'), 'Annotation %s containing some word not exist in dictionary!' % sample['VideoID'])
                        continue

                    summerize = (i - last_sum >= 100)
                    # take_checkpoint = (i != 0) and (i % 200 == 0)
                    print('Feed features into DNC.')

                    # reapeating same input for 'seq_reapte' times.
                    # for n in range(seq_reapte):
                    first_loss = None
                    n = 0
                    while True:
                        loss_value, out_value, _, summary = session.run([
                            loss,
                            softmax_output,
                            apply_gradients,
                            summerize_op if summerize else no_summerize,
                        ], feed_dict={
                            ncomputer.input_data: input_data,
                            ncomputer.target_output: target_outputs,
                            ncomputer.sequence_length: seq_len,
                            loss_weights: mask
                        })

                        # if loss_value == 0:
                        #     np.save('debug_target.npy', {'tar': target_outputs, 'out': out_value})
                        #     sys.exit(0)

                        print(colored('[%d]loss: ' % n, color='green'), loss_value)
                        n += 1
                        if first_loss is None:
                            first_loss = loss_value

                        if (n >= seq_reapte) or n >= seq_reapte * 10:
                            if not single_repeat and (not all(last_avg_min_max) or True):
                                break

                    last_100_losses.append(loss_value)
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

                    if i - last_log >= 500:
                        last_log = i
                        llprint("Saving Checkpoint ... "),
                        ncomputer.save(session, ckpts_dir, 'step-%d' % (i))
                        llprint("Done!")

                    input_data = target_outputs = seq_len = mask = None  # reset for next group of data.

                except KeyboardInterrupt:

                    llprint("Saving Checkpoint ... "),
                    ncomputer.save(session, ckpts_dir, 'step-%d' % (i))
                    llprint("Done!")
                    sys.exit(0)
