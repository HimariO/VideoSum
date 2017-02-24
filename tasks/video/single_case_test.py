import warnings
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
    output_len = 30

    from_checkpoint = None
    video_file = ''
    options, _ = getopt.getopt(sys.argv[1:], '', ['checkpoint=', 'video='])

    for opt in options:
        if opt[0] == '--checkpoint':
            from_checkpoint = opt[1]
        elif opt[0] == '--video':
            video_file = opt[1]

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
            softmax_output = tf.nn.softmax(output)

            llprint("Done!")

            llprint("Initializing Variables ... ")
            session.run(tf.global_variables_initializer())
            llprint("Done!")

            if from_checkpoint is not None:
                llprint("Restoring Checkpoint %s ... " % (from_checkpoint))
                ncomputer.restore(session, ckpts_dir, from_checkpoint)
                llprint("Done!")

            last_100_losses = []

            start_time_100 = time.time()
            end_time_100 = None
            avg_100_time = 0.
            avg_counter = 0

            try:

                # sample = np.random.choice(data, 1)
                try:
                    video_input = load_video(video_file)
                    seq_len = len(video_input) + output_len
                except:
                    print(colored('Error: ', color='red'), 'video %s doesn\'t exist.' % video_file)
                    sys.exit(0)

                print('Getting VGG features...')
                input_data = []
                for frame in video_input:
                    fc6 = session.run([vgg.fc6], feed_dict={image_holder: frame})
                    input_data.append(np.reshape(fc6, [-1]))

                for j in range(seq_len - len(input_data)):  # padding
                    input_data.append(np.zeros([input_size], dtype=np.float32))
                input_data = np.array([input_data])

                print('Feed features into DNC.')

                step_output = session.run([
                    softmax_output,
                ], feed_dict={
                    ncomputer.input_data: input_data,
                    ncomputer.sequence_length: seq_len,
                })

                sentence_output = ''
                word_map = ['' for i in range(len(lexicon_dict) + 1)]
                for word in lexicon_dict.keys():
                    ind = lexicon_dict[word]
                    word_map[ind] = word

                step_output = step_output[0]  # shape (1, n+30, 21866)
                N = step_output.shape[1]

                print(step_output.shape)
                print(step_output)

                for word in [step_output[:, i, :] for i in range(N)]:
                    index = np.argmax(word)
                    try:
                        sentence_output += word_map[index] + ' '
                    except:
                        print('Cant find in dictionary! ', index)

                print(colored('DCN: ', color='green'), sentence_output)

            except KeyboardInterrupt:

                llprint("Done!")
                sys.exit(0)
