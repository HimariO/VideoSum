import warnings
import tensorflow as tf
import numpy as np
import pickle
import getopt
import time
import sys
import os
import csv

from dnc.dnc import *
from VGG.vgg19 import Vgg19
from recurrent_controller import *
from post_controller import *
from train_until import *
from Visualize.CNNs_feat_distribution.get_single_videofeat import Extractor

from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image
from termcolor import colored
from tensorflow.python import debug as tf_db
from scipy import spatial

anno_file = './dataset/MSR_en.csv'
dict_file = './dataset/MSR_en_dict.csv'
video_dir = './dataset/YouTubeClips/'

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.ERROR)
    dirname = os.path.dirname(__file__)
    ckpts_dir = os.path.join(dirname, 'checkpoints')
    data_dir = os.path.join(dirname, 'data', 'en-10k')
    tb_logs_dir = os.path.join(dirname, 'test_logs')

    learning_rate = 1e-4
    momentum = 0.9
    output_len = 50

    from_checkpoint = None
    is_debug = False
    is_memview = False
    use_w2v = False
    use_VGG = False
    concat_inp = False
    device_choose = '/gpu:0'

    options, _ = getopt.getopt(sys.argv[1:], '', ['checkpoint=', 'debug=', 'memory_view=', 'word2vec=', 'device='])

    for opt in options:
        if opt[0] == '--checkpoint':
            from_checkpoint = opt[1]
        elif opt[0] == '--debug':
            lowerc = opt[1].lower()
            is_debug = lowerc == 't' or lowerc == 'true' or lowerc == '1'
        elif opt[0] == '--memory_view':
            lowerc = opt[1].lower()
            is_memview = lowerc == 't' or lowerc == 'true' or lowerc == '1'
        elif opt[0] == '--word2vec':
            lowerc = opt[1].lower()
            use_w2v = lowerc == 't' or lowerc == 'true' or lowerc == '1'
        elif opt[0] == '--device':
            lowerc = opt[1].lower()
            device_choose = lowerc

    llprint("Loading Data ... ")
    w2v_emb = None
    if use_w2v:
        dict_file = './dataset/MSR_enW2V_dict.csv'
        w2v_emb = np.load('./dataset/MSR_enW2V.npy') * 3
        tree = spatial.KDTree(w2v_emb)
    data, lexicon_dict = load(anno_file, dict_file)
    llprint("Done!\n")

    batch_size = 1
    input_size = 2048  # 100352
    output_size = len(lexicon_dict) if not use_w2v else w2v_emb.shape[1]
    sequence_max_length = 100
    word_space_size = len(lexicon_dict)
    words_count = 256
    word_size = 1024
    read_heads = 2

    graph = tf.Graph()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with graph.as_default():
        with tf.Session(graph=graph, config=config) as session:
            with tf.device(device_choose):

                if is_debug:
                    session = tf_db.LocalCLIDebugWrapperSession(session)
                    print(colored('Wrapping session with tfDebugger.', on_color='on_red'))

                print(colored('Running on [%s].' % device_choose, color='green'))
                llprint("Building Computational Graph ... ")
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
                #     testing=True
                # )
                ncomputer = DNCAuto(
                    AutoController,
                    input_size,
                    output_size,
                    sequence_max_length,
                    words_count,
                    word_size,
                    read_heads,
                    batch_size,
                    testing=True,
                    output_feedback=True
                )

                if use_VGG:
                    vgg = Vgg19(vgg19_npy_path='./VGG/vgg19.npy')
                    image_holder = tf.placeholder('float32', [1, 224, 224, 3])
                    vgg.build(image_holder)

                # summerizer = tf.summary.FileWriter(tb_logs_dir, session.graph)
                summaries = []

                output, memory_view, _ = ncomputer.get_outputs()
                memory_states = ncomputer.get_memoory_states()

                if not use_w2v:
                    softmax_output = tf.nn.softmax(output)
                else:
                    softmax_output = output

                llprint("Done!")

                if from_checkpoint is not None:
                    llprint("Restoring Checkpoint %s ... " % (from_checkpoint))
                    ncomputer.restore(session, ckpts_dir, from_checkpoint)
                    llprint("Done!")
                else:
                    llprint("Initializing Variables ... ")
                    session.run(tf.global_variables_initializer())
                    llprint("Done!")

                if not use_VGG:
                    llprint("Loading InceptionV3")
                    model = Extractor()

                last_100_losses = []

                start_time_100 = time.time()
                end_time_100 = None
                avg_100_time = 0.
                avg_counter = 0

                samples = np.random.choice(data[:100], 10)
                # samples = data[1689:1695]

                videos = ['%s_%s_%s.avi' % (f['VideoID'], f['Start'], f['End']) for f in samples]
                vid_targets = [f['Description'] for f in samples]
                if is_memview:
                    videos = ['Ugb_uH72d0I_8_17.avi']
                    # videos = ['eZLxohGP4IE_15_25.avi']

                weight_tFvars = []
                for var in tf.trainable_variables():
                    if "weight" in var.name:
                        weight_tFvars.append(var)

                for test_file, target in zip(videos, vid_targets):
                    try:
                        try:
                            video_input = load_video(video_dir + test_file, use_VGG=use_VGG, sample=6*2)
                            print(colored('frame count: ', color='yellow'), len(video_input))
                            seq_len = len(video_input) + output_len
                            input_len = len(video_input)
                        except:
                            print(colored('Error: ', color='red'), 'video %s doesn\'t exist.' % test_file)
                            continue
                            # sys.exit(0)

                        print('Getting CNN features...')

                        input_data = []

                        for frame in video_input:
                            if use_VGG:
                                fc6 = session.run([vgg.fc6], feed_dict={image_holder: frame})
                                input_data.append(np.reshape(fc6, [-1]))
                            else:
                                # print(frame)
                                feat = model.extract_PIL(Image.fromarray(frame))
                                input_data.append(np.reshape(feat, [-1]))

                        input_data = np.array(input_data)
                        if concat_inp:
                            input_data = inputConcat(input_data)

                        input_data = np.concatenate([input_data, np.zeros([seq_len - len(input_data), input_data.shape[1]])], axis=0)
                        input_data = np.array([input_data])  # add batch dimansion

                        if batch_size > 1:
                            input_data = np.concatenate([input_data for i in range(batch_size)], axis=0)

                        if input_data.shape[2] != ncomputer.input_size:
                            diff = abs(input_data.shape[2] - ncomputer.input_size)
                            input_data = np.concatenate([input_data, np.zeros([batch_size, input_data.shape[1], diff])], axis=2)
                            print(input_data.shape)

                        print('seqlen: ', seq_len)
                        print('Feed features into DNC.')

                        step_output, mem_tuple, mem_matrix = session.run([
                            softmax_output,
                            memory_view,
                            memory_states,
                        ], feed_dict={
                            ncomputer.input_data: input_data,
                            ncomputer.sequence_length: seq_len,
                        })

                        word_map = ['' for i in range(len(lexicon_dict) + 1)]
                        for word in lexicon_dict.keys():
                            ind = lexicon_dict[word]
                            word_map[ind] = word

                        sentence_output = ''
                        last_word = ''

                        sentence_indexs = []
                        top5_outputs = []

                        step_output = step_output[0]  # shape (1, n+30, 21866)
                        N = step_output.shape[0]

                        sentence_5 = [[''] for i in range(5)]

                        for word in [step_output[i, :] for i in range(N)]:
                            if use_w2v:
                                distances, index = tree.query(word, k=5)
                                top5_outputs.append([(word_map[i], d) for d, i in zip(distances, index)])
                            else:
                                index = np.argmax(word)
                                v = word.max()
                                _sorted = word.argsort()[-5:].tolist()
                                _sorted.reverse()

                                top5 = [(word_map[ID], word[ID]) for ID in _sorted]
                                top5_outputs.append(top5)

                        counter = 0

                        for t5 in top5_outputs:
                            for w, s in zip(t5, sentence_5):
                                if counter == input_len:
                                    s.append('[*]')
                                if s[-1] != w[0]:
                                    s.append(w[0])
                            counter += 1

                        print(colored('Target: ', color='cyan',), target)

                        for sent in sentence_5[:3]:
                            out = ''
                            for w in sent:
                                w = colored(w, color='green') if w in target else w
                                out += w + ' '
                            print(colored('DCN: ', color='green'), out)

                        # print(colored('DCN: ', color='green'), sentence_output)
                        # print(sentence_indexs)
                        # print(top5_outputs)
                        print('')
                        if is_memview:
                            # weights = []
                            # for var in weight_tFvars:
                            #     weights.append(session.run(var))
                            weights = session.run(weight_tFvars)
                            weights = {var.name: tensor for var, tensor in zip(weight_tFvars, weights)}

                            # np.save(test_file[:-4] + '_weights_%s.npy' % from_checkpoint, weights)
                            np.save(os.path.join('./Visualize', test_file[:-4] + '_memView_%s.npy' % from_checkpoint), mem_tuple)
                            np.save(os.path.join('./Visualize', test_file[:-4] + '_memMatrix_%s.npy' % from_checkpoint), mem_matrix)
                            np.save(os.path.join('./Visualize', test_file[:-4] + '_outputMatrix_%s.npy' % from_checkpoint), step_output)

                    except KeyboardInterrupt:

                        llprint("Done!")
                        sys.exit(0)
