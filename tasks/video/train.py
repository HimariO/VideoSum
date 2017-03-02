import warnings
import tensorflow as tf
import numpy as np
import pickle
import getopt
import time
import sys
import os
import csv
import spacy
import re

from dnc.dnc import DNC
from VGG.vgg19 import Vgg19
from recurrent_controller import RecurrentController, L2RecurrentController
from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image
from termcolor import colored

anno_file = './dataset/MSR_en.csv'
dict_file = './dataset/MSR_en_dict.csv'
video_dir = './dataset/YouTubeClips/'
nlp = spacy.load('en')


def load_video(filepath, sample=6):
    clip = VideoFileClip(filepath)
    video = []

    skip = 0
    for frame in clip.iter_frames():
        skip += 1
        if skip % sample != 0:
            continue

        img = Image.fromarray(frame)
        img = img.resize((224, 224))
        norm = np.divide(np.array(img), 255)
        norm = np.reshape(norm, [1, 224, 224, 3])
        video.append(norm)

    return np.array(video)


def llprint(message):
    print(colored(message, color='blue'))


def load(anno_path, dict_path):
    datas = []
    dictionary = {}

    with open(anno_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            datas.append(row)

    with open(dict_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            dictionary[row['word']] = int(row['id'])

    return datas, dictionary


def onehot(index, size):
    vec = np.zeros(size, dtype=np.float32)
    index = index if index < size else size - 1
    vec[index] = 1.0
    return vec


def prepare_sample(annotation, dictionary, video_feature, redu_sample_rate=1):
    file_path = video_dir + '%s_%s_%s.avi' % (annotation['VideoID'], annotation['Start'], annotation['End'])
    # some video in annotation does not include in dataset.
    if not os.path.isfile(file_path):
        raise OSError(2, 'No such file or directory', file_path)
    elif type(video_feat) is not np.ndarray:
        if video_feature == 0 or video_feature == [0]:
            raise OSError(2, 'Empty data slot!', file_path)

    video_feature = [np.reshape(i, [-1]) for i in video_feature]
    input_vec = np.array(video_feature[::redu_sample_rate])

    output_str = [str(i) for i in nlp(annotation['Description'][:-5])] + ['<EOS>']
    output_str = [dictionary[i] for i in output_str]
    seq_len = input_vec.shape[0] + len(output_str) + 1

    output_str = [np.zeros(word_space_size) for i in range(input_vec.shape[0] + 1)] + [onehot(i, word_space_size) for i in output_str]
    output_vec = np.array(output_str, dtype=np.float32)

    mask = np.zeros(seq_len, dtype=np.float32)
    mask[input_vec.shape[0]:] = 1

    padding_i = np.zeros([seq_len - input_vec.shape[0], 4096], dtype=np.float32)
    input_vec = np.concatenate([input_vec, padding_i], axis=0)

    print(colored('seq_len: ', color='yellow'), seq_len)
    # print(colored('input_vec: ', color='yellow'), input_vec.shape)
    # print(colored('output_vec: ', color='yellow'), output_vec.shape)
    return (
        input_vec,
        np.reshape(output_vec, (1, -1, word_space_size)),
        seq_len,
        np.reshape(mask, (1, -1, 1))
    )


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
    data, lexicon_dict = load(anno_file, dict_file)
    llprint("Done!\n")

    batch_size = 1
    input_size = 4096
    output_size = len(lexicon_dict)
    sequence_max_length = 100
    word_space_size = len(lexicon_dict)
    words_count = 256
    word_size = 128
    read_heads = 4

    learning_rate = 1e-4
    momentum = 0.9

    from_checkpoint = None
    iterations = len(data)
    start_step = 0

    last_sum = 1
    last_log = 1
    mis_data_offset = 0

    options, _ = getopt.getopt(sys.argv[1:], '', ['checkpoint=', 'iterations=', 'start='])

    """
    If need to resume training from checkpoint, you must start form iteration which have same step with one of npy sample's start file.
    due to teh fact that sample in the middle of file have unknow ID(start + miss_data_num), after program restarting with 0 miss_data_num.
    """

    for opt in options:
        if opt[0] == '--checkpoint':
            from_checkpoint = opt[1]
        elif opt[0] == '--iterations':
            iterations = int(opt[1])
        elif opt[0] == '--start':
            last_sum = last_log = start_step = int(opt[1])

    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(graph=graph) as session:

            llprint("Building Computational Graph ... ")
            # llprint("Building VGG ... ")
            #
            # vgg = Vgg19(vgg19_npy_path='./VGG/vgg19.npy')
            # image_holder = tf.placeholder('float32', [1, 224, 224, 3])
            # vgg.build(image_holder)

            llprint("Done!")
            llprint("Building DNC ... ")

            optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=momentum)
            summerizer = tf.summary.FileWriter(tb_logs_dir, session.graph)

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

            output, _ = ncomputer.get_outputs()

            loss_weights = tf.placeholder(tf.float32, [batch_size, None, 1])
            # output tensors will containing all output from both input steps and output steps.
            loss = tf.reduce_mean(
                loss_weights * tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=ncomputer.target_output)
            )

            summeries = []

            gradients = optimizer.compute_gradients(loss)
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    gradients[i] = (tf.clip_by_value(grad, -10, 10), var)
            for (grad, var) in gradients:
                if grad is not None:
                    summeries.append(tf.summary.histogram(var.name + '/grad', grad))

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

            last_100_losses = []

            start = 0 if start_step == 0 else start_step + 1
            end = start_step + iterations + 1 if start_step + iterations + 1 < len(data) else len(data)

            start_time_100 = time.time()
            end_time_100 = None
            avg_100_time = 0.
            avg_counter = 0

            current_feat = (None, -1, -1)  # [npy, start_id, end_id]
            input_data = target_outputs = seq_len = mask = None
            included_vid = 0
            seq_reapte = 3

            # i = start
            for i in range(start, end):
                if current_feat is None or current_feat[1] > i or current_feat[2] < i:
                    for f in feat_files_tup:
                        if f[1] <= i and f[2] >= i:
                            llprint('Loading %s...' % f[0])
                            current_feat = np.load('./dataset/' + f[0]), f[1], f[2]
                            mis_data_offset = 0
                            # i = f[1]
                            break
                        else:
                            current_feat = (None, -1, -1)
                    if current_feat == (None, -1, -1):  # if no data are aliviable.
                        print('Cant find suitable data for tarining.')
                        sys.exit(0)

                try:
                    llprint("\rIteration %d/%d" % (i, end))

                    # sample = np.random.choice(data, 1)
                    sample = data[i]
                    video_feat = current_feat[0][i - current_feat[1]]
                    try:
                        input_data_, target_outputs_, seq_len_, mask_ = prepare_sample(sample, lexicon_dict, video_feat, redu_sample_rate=2)

                        input_data = np.concatenate([input_data, input_data_], axis=0) if input_data is not None else input_data_
                        target_outputs = np.concatenate([target_outputs, target_outputs_], axis=1) if target_outputs is not None else target_outputs_
                        seq_len = seq_len + seq_len_ if seq_len is not None else seq_len_
                        mask = np.concatenate([mask, mask_], axis=1) if mask is not None else mask_
                        included_vid += 1

                        # i += 1
                        if included_vid < 10:
                            continue
                        else:
                            included_vid = 0
                    except OSError:
                        print(colored('Error: ', color='red'), 'video %s doesn\'t exist.' % sample['VideoID'])
                        # mis_data_offset += 1
                        # i += 1
                        continue
                    except KeyError:
                        print(colored('Error: ', color='red'), 'Annotation %s containing some word not exist in dictionary!' % sample['VideoID'])
                        # mis_data_offset += 1
                        # i += 1
                        continue

                    summerize = (i % 100 == 0)
                    take_checkpoint = (i != 0) and (i % 200 == 0)
                    print('Feed features into DNC.')

                    # reapeating same input for 'seq_reapte' times.
                    # for n in range(seq_reapte):
                    first_loss = None
                    n = 0
                    while True:
                        loss_value, _, summary = session.run([
                            loss,
                            apply_gradients,
                            summerize_op if summerize else no_summerize
                        ], feed_dict={
                            ncomputer.input_data: np.array([input_data]),
                            ncomputer.target_output: target_outputs,
                            ncomputer.sequence_length: seq_len,
                            loss_weights: mask
                        })

                        print(colored('loss: ', color='yellow'), loss_value)
                        n += 1
                        if first_loss is None:
                            first_loss = loss_value
                        elif (loss_value < first_loss and n >= seq_reapte) or n >= seq_reapte * 10:
                            break

                    last_100_losses.append(loss_value)
                    summerizer.add_summary(summary, i)

                    if i - last_sum >= 100:
                        last_sum = i
                        llprint("   Avg. Cross-Entropy: %.7f" % (np.mean(last_100_losses)))
                        llprint("   Max. %.7f  Min. %.7f" % (max(last_100_losses), min(last_100_losses)))

                        end_time_100 = time.time()
                        elapsed_time = (end_time_100 - start_time_100) / 60
                        avg_counter += 1
                        avg_100_time += (1. / avg_counter) * (elapsed_time - avg_100_time)
                        estimated_time = (avg_100_time * ((end - i) / 100.)) / 60.

                        print("Avg. 100 iterations time: %.2f minutes" % (avg_100_time))
                        print("Approx. time to completion: %.2f hours" % (estimated_time))

                        start_time_100 = time.time()
                        last_100_losses = []

                    if i - last_log >= 200:
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
