import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import numpy as np
import pickle
import getopt
import time
import sys
import os
import csv

from dnc.dnc import DNC
from VGG.vgg19 import Vgg19
from recurrent_controller import RecurrentController
from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image
from termcolor import colored

anno_file = './dataset/MSR_en.csv'
dict_file = './dataset/MSR_en_dict.csv'
video_dir = './dataset/YouTubeClips/'
lastvideo = {'path': '', 'seq_len': 0, 'features': None, 'video': None}
skip_vgg = False


def load_video(filepath, sample=3):
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
    vec[index] = 1.0
    return vec


def prepare_sample(annotation, dictionary):
    file_path = video_dir + '%s_%s_%s.avi' % (annotation['VideoID'], annotation['Start'], annotation['End'])
    if lastvideo['path'] != file_path:
        if os.path.isfile(file_path):
            input_vec = load_video(file_path)
            lastvideo['path'] = file_path
            lastvideo['video'] = input_vec
            skip_vgg = False
        else:
            raise OSError(2, 'No such file or directory', file_path)
    else:
        skip_vgg = True
        input_vec = lastvideo['video']

    output_str = annotation['Description'].split()
    output_str = [dictionary[i] for i in output_str]
    seq_len = input_vec.shape[0] + len(output_str) + 1

    output_str = [np.zeros(word_space_size) for i in range(input_vec.shape[0] + 1)] + [onehot(i, word_space_size) for i in output_str]
    output_vec = np.array(output_str, dtype=np.float32)

    print(colored('seq_len: ', color='yellow'), seq_len)
    print(colored('input_vec: ', color='yellow'), input_vec.shape)
    print(colored('output_vec: ', color='yellow'), output_vec.shape)
    return (
        input_vec,
        np.reshape(output_vec, (1, -1, word_space_size)),
        seq_len,
    )


if __name__ == '__main__':

    dirname = os.path.dirname(__file__)
    ckpts_dir = os.path.join(dirname, 'checkpoints')
    data_dir = os.path.join(dirname, 'data', 'en-10k')
    tb_logs_dir = os.path.join(dirname, 'logs')

    llprint("Loading Data ... ")
    data, lexicon_dict = load(anno_file, dict_file)
    llprint("Done!\n")

    batch_size = 1
    input_size = 4096  # 100352
    output_size = len(lexicon_dict)
    sequence_max_length = 100
    word_space_size = len(lexicon_dict)
    words_count = 256
    word_size = 64
    read_heads = 4

    learning_rate = 1e-5
    momentum = 0.9

    from_checkpoint = None
    iterations = len(data)
    start_step = 0

    options, _ = getopt.getopt(sys.argv[1:], '', ['checkpoint=', 'iterations=', 'start='])

    for opt in options:
        if opt[0] == '--checkpoint':
            from_checkpoint = opt[1]
        elif opt[0] == '--iterations':
            iterations = int(opt[1])
        elif opt[0] == '--start':
            start_step = int(opt[1])

    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(graph=graph) as session:

            llprint("Building Computational Graph ... ")
            llprint("Building VGG ... ")

            vgg = Vgg19(vgg19_npy_path='./VGG/vgg19.npy')
            image_holder = tf.placeholder('float32', [1, 224, 224, 3])
            vgg.build(image_holder)

            llprint("Done!")
            llprint("Building DNC ... ")

            optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=momentum)
            summerizer = tf.summary.FileWriter(tb_logs_dir, session.graph)

            ncomputer = DNC(
                RecurrentController,
                input_size,
                output_size,
                sequence_max_length,
                words_count,
                word_size,
                read_heads,
                batch_size
            )

            output, _ = ncomputer.get_outputs()

            # loss_weights = tf.placeholder(tf.float32, [batch_size, None, 1])
            # output tensors will containing all output from both input steps and output steps.
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=ncomputer.target_output)
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

            for i in range(start, end + 1):
                try:
                    llprint("\rIteration %d/%d" % (i, end))

                    # sample = np.random.choice(data, 1)
                    sample = data[i]
                    try:
                        video_input, target_outputs, seq_len = prepare_sample(sample, lexicon_dict)
                    except:
                        print(colored('Error: ', color='red'), 'video %s doesn\'t exist.' % sample['VideoID'])
                        continue

                    print('Getting VGG features...')

                    input_data = []
                    if not skip_vgg:
                        for frame in video_input:
                            fc6 = session.run([vgg.fc6], feed_dict={image_holder: frame})
                            input_data.append(np.reshape(fc6, [-1]))

                        for j in range(seq_len - len(input_data)):  # padding
                            input_data.append(np.zeros([input_size], dtype=np.float32))
                        lastvideo['features'] = input_data = np.array([input_data])
                    else:
                        input_data = lastvideo['features']

                    summerize = (i % 100 == 0)
                    take_checkpoint = (i != 0) and (i % 200 == 0)
                    print('Feed features into DNC.')

                    loss_value, _, summary = session.run([
                        loss,
                        apply_gradients,
                        summerize_op if summerize else no_summerize
                    ], feed_dict={
                        ncomputer.input_data: input_data,
                        ncomputer.target_output: target_outputs,
                        ncomputer.sequence_length: seq_len,
                    })

                    print(colored('loss: ', color='yellow'), loss_value)
                    last_100_losses.append(loss_value)
                    summerizer.add_summary(summary, i)

                    if summerize:
                        llprint("   Avg. Cross-Entropy: %.7f" % (np.mean(last_100_losses)))

                        end_time_100 = time.time()
                        elapsed_time = (end_time_100 - start_time_100) / 60
                        avg_counter += 1
                        avg_100_time += (1. / avg_counter) * (elapsed_time - avg_100_time)
                        estimated_time = (avg_100_time * ((end - i) / 100.)) / 60.

                        print("Avg. 100 iterations time: %.2f minutes" % (avg_100_time))
                        print("Approx. time to completion: %.2f hours" % (estimated_time))

                        start_time_100 = time.time()
                        last_100_losses = []

                    if take_checkpoint:
                        llprint("Saving Checkpoint ... "),
                        ncomputer.save(session, ckpts_dir, 'step-%d' % (i))
                        llprint("Done!")

                except KeyboardInterrupt:

                    llprint("Saving Checkpoint ... "),
                    ncomputer.save(session, ckpts_dir, 'step-%d' % (i))
                    llprint("Done!")
                    sys.exit(0)
