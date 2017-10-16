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
import math
import json

from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image
from termcolor import colored
from sklearn import preprocessing
from scipy import spatial

# TODO: try to putting togather all label, annotation, map file.
anno_file = './dataset/MSR_en.csv'
dict_file = './dataset/MSR_en_dict.csv'
w2v_dict_file = './dataset/MSR_enW2V_dict.csv'
video_dir = './dataset/YouTubeClips/'
word2v_emb_file = './dataset/MSR_enW2V.npy'
nlp = spacy.load('en')
ham_map = None
mul_onehot_map = None


def onehot_vec2id(nparr):
    result = []
    for batch in nparr:
        batch_vec = []
        for step in batch:
            batch_vec.append(step.argmax())
        result.append(batch_vec)
    return np.array(result)


def decode_output(lexicon_dict, step_output, target='[?]', word2v_emb=None, hamming=None, mul_onehot=None):

    assert type(step_output) is np.ndarray
    assert len(step_output.shape) == 2  # [time_step, output_vector]

    word_map = ['' for i in range(len(lexicon_dict) + 1)]
    if word2v_emb is not None:
        tree = spatial.KDTree(word2v_emb)

    for word in lexicon_dict.keys():
        ind = lexicon_dict[word]
        word_map[ind] = word

    sentence_5 = [[' '] for i in range(5)]
    top5_outputs = []

    N = step_output.shape[0]

    global mul_onehot_map

    if mul_onehot_map is not None:
        mul_onehot_map = mul_onehot_remap(config=mul_onehot)
        tree = spatial.KDTree(get_mul_onehot_sapce(lexicon_dict, mul_onehot))

    map_inverse = {}

    for i in range(len(mul_onehot_map)):
        map_inverse[mul_onehot_map[i]] = i

    for word in [step_output[i, :] for i in range(N)]:
        if mul_onehot is not None:
            # TODO: add argmax decode back.

            if False:
                # index of word vector in npy file are order by word id, so no need to reverse remaped id.
                distances, index = tree.query(word, k=5)
                top5_outputs.append([(word_map[i], d) for d, i in zip(distances, index)])
            else:
                fake_batch = word.reshape([1, 1, word.shape[0]])
                pred_ids = mul_onehot2ids(fake_batch)[0][0]  # (1, 1, mul_onehot[1])
                word_id = 0
                for k in range(mul_onehot[1]):
                    word_id += pred_ids[k] * mul_onehot[0]**k
                word_id = map_inverse[word_id] if map_inverse[word_id] <= len(word_map) else 2
                top5_outputs.append([(word_map[word_id], word_id)] * 5)

        elif word2v_emb is not None:
            distances, index = tree.query(word, k=5)
            top5_outputs.append([(word_map[i], d) for d, i in zip(distances, index)])
        else:
            index = np.argmax(word)
            v = word.max()
            _sorted = word.argsort()[-5:].tolist()
            _sorted.reverse()
            # print(word_map[index])

            top5 = [(word_map[ID], word[ID]) for ID in _sorted]
            top5_outputs.append(top5)

    counter = 0
    # print(top5_outputs, '\n', sentence_5)
    for t5 in top5_outputs:
        for w, s in zip(t5, sentence_5):
            # if counter == input_len:
            #     s.append('[*]')
            # print(w, s)
            if s[-1] != w[0]:
                s.append(w[0])
        counter += 1

    # print(colored('Target: ', color='cyan',), target)

    output_sentence = []
    for sent in sentence_5:
        out = ''
        for w in sent:
            w = colored(w, color='green') if w in target else w
            out += w + ' '
        output_sentence.append(out)
    return output_sentence


def get_video_name(annotation):
    # print(annotation)
    return '%s_%s_%s' % (annotation['VideoID'], annotation['Start'], annotation['End'])


def load_video(filepath, sample=6, use_VGG=True):
    clip = VideoFileClip(filepath)
    video = []

    skip = 0
    for frame in clip.iter_frames():
        skip += 1
        if skip % sample != 0:
            continue

        img = Image.fromarray(frame)
        img = img.resize((224, 224)) if use_VGG else img.resize((299, 299))
        if use_VGG:
            norm = np.divide(np.array(img), 255)
            norm = np.reshape(norm, [1, 224, 224, 3])
            video.append(norm)
        else:
            # keras will handle input normalization for InceptionV3
            video.append(np.array(img))

    return np.array(video)


def llprint(message):
    print(colored(message, color='blue'))


def load(anno_path, dict_path):
    datas = []
    dictionary = {'': 0}

    with open(anno_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            datas.append(row)

    with open(dict_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            dictionary[row['word']] = int(row['id'])

    return datas, dictionary


def load_json(json_path):
    with open(json_path, mode='r') as json_file:
        return json.load(json_file)


def onehot(index, size):
    vec = np.zeros(size, dtype=np.float32)
    index = index if index < size else size - 1
    vec[index] = 1.0
    return vec


def id2hamming(index, bits, map_path='./dataset/hamming_map.npy'):
    """
    translate word index into fixed length hamming code:
        binayr code with fix length, same hamming distance between every code pair from the same set.

    bits: length of code, if bits=6, output code will be permutaion of [0,0,0,1,1,1]
          ex:  [0,1,1,0,1,0]    [1,0,1,0,1,0]    [1,1,0,0,0,1]   ....
    """
    assert bits % 2 == 0

    def index2bin(ids):
        base = np.zeros([bits])
        for id in ids:
            base[id] = 1
        return base

    def H_mtx(n):
        H2 = [
            [-1, 1, 1],
            [1, 1, -1],
            [1, -1, 1],
            [1, -1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
        ]
        if n == 2:
            return np.array(H2)
        else:
            h_mtx = H_mtx(n // 2)
            top_h = np.concatenate([h_mtx, h_mtx], axis=1)
            btn_h = np.concatenate([-h_mtx, h_mtx], axis=1)
            h = np.concatenate([top_h, btn_h], axis=0)
            return h

    def gen():
        print('Generating hamming code~')
        import itertools as it
        import random

        perum = list(it.combinations(range(bits), 4))
        perum = list(map(lambda x: index2bin(x), perum))
        random.shuffle(perum)
        # perum = H_mtx(bits)
        # perum = np.concatenate([-perum, perum], axis=0)
        # perum = (perum + 1) / 2
        np.save(map_path, np.array(perum))
        print('Save result to %s' % map_path)

    if not os.path.exists(map_path):
        gen()

    global ham_map

    if ham_map is None:
        ham_map = np.load(map_path)

    # if ham_map.shape[1] != bits:
    #     gen()
    #     ham_map = np.load(map_path)

    max_num = len(ham_map)
    # assert index < max_num
    return ham_map[index]


def mul_onehot_remap(path='./dataset/mul_oneshot_remap.json', config=(256, 2)):
    import json
    import random

    if os.path.exists(path):
        with open(path, mode='r') as J:
            return json.load(J)
    else:
        all_n = list(range(config[0] ** config[1]))
        random.shuffle(all_n)
        with open(path, mode='w') as J:
            print("Create new MultiOneHot Mapping file to %s" % path)
            json.dump(all_n, J)
            return all_n


def id2mul_onehot(index, config=(256, 2)):
    global mul_onehot_map

    if mul_onehot_map is None:
        mul_onehot_map = mul_onehot_remap(config=config)

    index = mul_onehot_map[index]
    mul_index = [(index % config[0]**i) // config[0]**(i - 1) for i in range(1, 1 + config[1])]
    # print(mul_index)
    result = None
    for i in mul_index:
        if result is None:
            result = onehot(i, config[0])
        else:
            result = np.concatenate([result, onehot(i, config[0])], axis=0)
    return result


def get_mul_onehot_sapce(dictionary, config, path='./dataset/mul_onehot_space.npy'):
    if os.path.exists(path):
        return np.load(path)
    else:
        vec_space = [np.zeros([config[0] * config[1]])] * (len(dictionary) + 1)
        for word in dictionary.keys():
            word_id = dictionary[word]
            word_vec = id2mul_onehot(word_id)
            vec_space[word_id] = word_vec
        vec_space = np.array(vec_space)
        np.save(path, vec_space)
        return vec_space


def mul_onehot2ids(vec, config=(256, 2)):
    result = []
    for batch in vec:
        b_ids = []
        for step in batch:
            s_ids = []
            for i in range(config[1]):
                st = i * config[0]
                ed = (i + 1) * config[0]
                s_ids.append(step[st:ed].argmax())
            b_ids.append(s_ids)
        result.append(b_ids)
    return np.array(result)


def put_bucket(seq_data):
    length_options = [20, 50, 100, 150]
    data_len = seq_data.shape[0] if len(seq_data.shape) <= 2 else seq_data.shape[1]
    bucket_size = -1

    for i in range(0, len(length_options)):
        if data_len <= length_options[i]:
            bucket_size = length_options[i]
            break

    padding_shape = list(seq_data.shape)
    padding_shape[-2] = bucket_size - data_len
    padding = np.zeros(padding_shape)

    seq_data = np.concatenate([padding, seq_data], axis=len(seq_data.shape) - 2)
    return seq_data


def prepare_sample(annotation, dictionary, video_feature, redu_sample_rate=1,
                   word_emb=None, hamming=None, mul_onehot=None, extend_target=False, setMask=True, concat_input=False, norm=False, bucket=False):
    EOS = '<EOS>' if word_emb is None else 'EOS'
    # choice output_szie depend on embedding method. (ont-hot or word2vec)
    if hamming is not None:
        word_space_size = hamming
    elif mul_onehot is not None:
        word_space_size = mul_onehot[0] * mul_onehot[1]
    elif word_emb is not None:
        word_space_size = word_emb.shape[1]
    else:
        word_space_size = len(dictionary)

    # some video in annotation does not included in dataset.
    # if not os.path.isfile(file_path):
    #     raise OSError(2, 'No such file or directory', file_path)
    if type(video_feature) is not np.ndarray:
        if video_feature == 0 or video_feature == [0]:
            file_path = video_dir + '%s_%s_%s.avi' % (annotation['VideoID'], annotation['Start'], annotation['End'])
            raise OSError(2, 'Empty data slot!', file_path)

    video_feature = [np.reshape(i, [-1]) for i in video_feature]
    input_vec = np.array(video_feature[::redu_sample_rate])

    if bucket:
        input_vec = put_bucket(input_vec)
        # output_vec = put_bucket(output_vec)
        # mask = put_bucket(mask)
        # seq_len = input_vec.shape[0]

    if concat_input:
        prev_input = np.array(video_feature[::redu_sample_rate][1:])
        prev_input = np.concatenate([np.zeros([1, input_vec.shape[1]]), prev_input], axis=0)
        input_vec = np.concatenate([input_vec, prev_input], axis=1)

    output_str = [str(i) for i in nlp(annotation['Description'][:-5])] + [EOS]
    dictionary_k = dictionary.keys()
    temp = []
    for word in output_str:
        try:
            temp.append(dictionary[word])
        except:
            temp.append(2)  # <UNKNOW>
    output_str = temp

    if extend_target:
        temp = []
        for s in output_str:
            for _ in range(3):
                temp.append(s)
        output_str = temp

    print('video_seq: %d, target_seq: %d' % (input_vec.shape[0], len(output_str)))
    input_len, output_len = input_vec.shape[0], len(output_str)

    seq_len = input_vec.shape[0] + len(output_str)
    output_vec = [np.zeros(word_space_size) for i in range(input_vec.shape[0])]

    if word_emb is not None:
        output_vec += [word_emb[i] for i in output_str]
    elif hamming is not None:
        output_vec += [id2hamming(i, hamming) for i in output_str]
    elif mul_onehot is not None:
        output_vec += [id2mul_onehot(i, config=mul_onehot) for i in output_str]
    else:
        output_vec += [onehot(i, word_space_size) for i in output_str]

    output_vec = np.array(output_vec, dtype=np.float32)

    mask = np.zeros(seq_len, dtype=np.float32)
    if setMask:
        mask[input_vec.shape[0]:] = 1
    else:
        mask[:] = 1

    if seq_len > input_vec.shape[0]:
        padding_i = np.zeros([seq_len - input_vec.shape[0], input_vec.shape[1]], dtype=np.float32)
        input_vec = np.concatenate([input_vec, padding_i], axis=0)

    if norm:
        preprocessing.normalize(input_vec, axis=1)
        # preprocessing.scale(input_vec, axis=1)

    mask = np.reshape(mask, (1, -1, 1))

    print(colored('seq_len: ', color='yellow'), seq_len)
    # print(colored('input_vec: ', color='yellow'), input_vec.shape)
    # print(colored('output_vec: ', color='yellow'), output_vec.shape)
    return (
        np.reshape(input_vec, (1, -1, input_vec.shape[1])),
        np.reshape(output_vec, (1, -1, word_space_size)),
        {'seq_len': seq_len, 'input_len': input_len, 'output_len': output_len},
        mask
    )


def prepare_mixSample(annotation, dictionary, video_feature, redu_sample_rate=1,
                      word_emb=None, hamming=None, extend_target=False, setMask=True, post_padding=0, concat_input=False, norm=False, bucket=True):
    """
    output input-output pair with overlay timestep.
    input_vector = [D, D, D, D, D, 0, 0, 0, 0]
    output_vector= [0, 0, 0, D, D, D, D, D, D]

    return:
    input_data (batch_size, seq_len, input_size)
    target_output (batch_size, seq_len, output_size)
    seq_len (1) = input_len + output_len - over_lap
    mask ()
    """

    file_path = video_dir + '%s_%s_%s.avi' % (annotation['VideoID'], annotation['Start'], annotation['End'])
    EOS = '<EOS>' if word_emb is None else 'EOS'

    if hamming is not None:
        word_space_size = hamming
    elif word_emb is not None:
        word_space_size = word_emb.shape[1]
    elif mul_onehot is not None:
        word_space_size = mul_onehot[0] * mul_onehot[1]
    else:
        word_space_size = len(dictionary)

    # some video in annotation does not include in dataset.
    if not os.path.isfile(file_path):
        raise OSError(2, 'No such file or directory', file_path)
    elif type(video_feature) is not np.ndarray:
        if video_feature == 0 or video_feature == [0]:
            raise OSError(2, 'Empty data slot!', file_path)

    video_feature = [np.reshape(i, [-1]) for i in video_feature]
    input_vec = np.array(video_feature[::redu_sample_rate])

    if bucket:
        input_vec = put_bucket(input_vec)
        # output_vec = put_bucket(output_vec)
        # mask = put_bucket(mask)
        # seq_len = input_vec.shape[0]

    if concat_input:
        prev_input = np.array(video_feature[::redu_sample_rate][1:])
        prev_input = np.concatenate([np.zeros([1, input_vec.shape[1]]), prev_input], axis=0)
        input_vec = np.concatenate([input_vec, prev_input], axis=1)

    output_str = [str(i) for i in nlp(annotation['Description'][:-5])] + [EOS]
    dictionary_k = dictionary.keys()
    temp = []
    for word in output_str:
        try:
            temp.append(dictionary[word])
        except:
            temp.append(2)  # <UNKNOW>
    output_str = temp

    if extend_target:
        temp = []
        for s in output_str:
            for _ in range(3):
                temp.append(s)
        output_str = temp

    print('video_seq: %d, target_seq: %d' % (input_vec.shape[0], len(output_str)))
    input_len, output_len = input_vec.shape[0], len(output_str)

    over_lap = input_len // 2 if min([len(output_str), input_vec.shape[0]]) >= input_len // 2 else min([len(output_str), input_vec.shape[0]])

    seq_len = input_vec.shape[0] + len(output_str) - over_lap

    output_vec = [np.zeros(word_space_size) for i in range(input_vec.shape[0] - over_lap)]
    if word_emb is not None:
        output_vec += [word_emb[i] for i in output_str]
    elif hamming is not None:
        output_vec += [id2hamming(i, hamming) for i in output_str]
    elif mul_onehot is not None:
        output_vec += [id2mul_onehot(i, config=mul_onehot) for i in output_str]
    else:
        output_vec += [onehot(i, word_space_size) for i in output_str]

    output_vec = output_vec + [np.zeros(word_space_size) for i in range(seq_len - len(output_vec))]
    output_vec = np.array(output_vec, dtype=np.float32)

    mask = np.zeros(seq_len, dtype=np.float32)
    mask[-len(output_str):] = 1

    if seq_len > input_vec.shape[0]:
        padding_i = np.zeros([seq_len - input_vec.shape[0], input_vec.shape[1]], dtype=np.float32)
        input_vec = np.concatenate([input_vec, padding_i], axis=0)

    if post_padding > 0:
        output_vec = np.concatenate([output_vec, np.zeros([post_padding, word_space_size])], axis=0)
        input_vec = np.concatenate([input_vec, np.zeros([post_padding, 2048])], axis=0)
        mask = np.concatenate([mask, np.zeros(post_padding)])
        seq_len += post_padding

    if norm:
        preprocessing.normalize(input_vec, axis=1)
        # preprocessing.scale(input_vec, axis=1)

    mask = np.reshape(mask, (1, -1, 1))

    print(colored('seq_len: ', color='yellow'), seq_len)
    # print(colored('input_vec: ', color='yellow'), input_vec.shape)
    # print(colored('output_vec: ', color='yellow'), output_vec.shape)
    return (
        np.reshape(input_vec, (1, -1, input_vec.shape[1])),
        np.reshape(output_vec, (1, -1, word_space_size)),
        {'seq_len': seq_len, 'input_len': input_len, 'output_len': output_len},
        mask
    )


def prepare_batch(annotations, dictionary, video_features, epoch, redu_sample_rate=1,
                  word_emb=None, hamming=None, mul_onehot=None, extend_target=False, setMask=True, useMix=False, accuTrain=False):
    """
    Put batch of label and features into a single nparray.
    annotations: list of annotations with size of 'batch_size'
    video_features: list of featrues with size of 'batch_size'

    Input will became:
    [
        [0, 0, I, I, I, 0, 0, 0 ,0],  Each row is single input sequce pair with lable output.
        [I, I, I, I, I, 0, 0, 0 ,0],
        [0, 0, 0, I, I, 0, 0, 0 ,0],
    ]

    Output will became:
    [
        [0, 0, 0, 0, 0, Y, 0, 0 ,0],
        [0, 0, 0, 0, 0, Y, Y, Y ,0],
        [0, 0, 0, 0, 0, Y, Y, Y ,Y],
    ]
    """

    input_data = None
    target_outputs = None
    seq_len = None
    mask = None

    word_space_size = len(dictionary) if word_emb is None else word_emb.shape[1]
    word_space_size = len(dictionary) if hamming is None else hamming
    none_padded = []

    for video_feature, annotation in zip(video_features, annotations):
        try:
            if useMix:
                input_data_, target_outputs_, seq_len_, mask_ = prepare_mixSample(annotation, dictionary, video_feature, redu_sample_rate, word_emb, hamming, extend_target, setMask)
            else:
                input_data_, target_outputs_, seq_len_, mask_ = prepare_sample(annotation, dictionary, video_feature, redu_sample_rate, word_emb, hamming, extend_target, setMask)

            if accuTrain:
                input_data_, target_outputs_, seq_len_, mask_ = accuTraining(input_data_, target_outputs_, mask_, seq_len_, epoch)

        except OSError:
            input_data_ = np.zeros([1, 2, 2048])
            target_outputs_ = np.zeros([1, 2, word_space_size])
            seq_len_ = {'seq_len': 2, 'input_len': 1, 'output_len': 1}
            mask_ = np.zeros([1, 2, 1])

        none_padded.append([input_data_, target_outputs_, seq_len_, mask_])

    max_in = max(none_padded, key=lambda x: x[2]['input_len'])[2]['input_len']
    max_out = max(none_padded, key=lambda x: x[2]['output_len'])[2]['output_len']

    max_out_step = list(map(lambda x: x[2]['seq_len'] - x[2]['output_len'], none_padded))
    max_out_step = max(max_out_step)
    seq_len = max_seq = max_in + max_out

    for data in none_padded:
        input_data_, target_outputs_, seq_len_, mask_ = data[0], data[1], data[2], data[3]

        if useMix:
            out_step = seq_len_['seq_len'] - seq_len_['output_len']
            left_pad = max_out_step - out_step
            # if seq_len_['output_len'] > 5:
            right_pad = max_out - seq_len_['output_len']

        else:
            left_pad = max_in - seq_len_['input_len']
            right_pad = max_out - seq_len_['output_len']

        input_data_ = np.concatenate([np.zeros([1, left_pad, 2048]), input_data_, np.zeros([1, right_pad, 2048])], axis=1)
        target_outputs_ = np.concatenate([np.zeros([1, left_pad, word_space_size]), target_outputs_, np.zeros([1, right_pad, word_space_size])], axis=1)

        mask_ = np.concatenate(
            [
                np.zeros([1, left_pad, 1]),
                mask_,
                np.zeros([1, right_pad, 1]),
            ],
            axis=1
        )
        # print(colored('mask_: ', color='yellow'), mask_.shape, ' ', left_pad, ' ', right_pad)
        # print(colored('input_vec: ', color='yellow'), input_data_.shape)
        # print(colored('output_vec: ', color='yellow'), target_outputs_.shape)

        input_data = np.concatenate([input_data, input_data_], axis=0) if input_data is not None else input_data_
        target_outputs = np.concatenate([target_outputs, target_outputs_], axis=0) if target_outputs is not None else target_outputs_
        mask = np.concatenate([mask, mask_], axis=0) if mask is not None else mask_

    if useMix:
        seq_len = input_data.shape[1]
    # np.save("debug.npy", {'in': input_data, 'out': target_outputs})
    # input("stop here.")
    return (
        input_data,
        target_outputs,
        {'seq_len': seq_len},
        mask
    )


def inputConcat(input_vec):
    prev_input = input_vec[1:]
    prev_input = np.concatenate([np.zeros([1, input_vec.shape[1]]), prev_input], axis=0)
    print(prev_input.shape)
    print(input_vec.shape)
    input_vec = np.concatenate([input_vec, prev_input], axis=1)
    print(input_vec.shape)
    return input_vec


def accuTraining(input_data_, target_outputs_, mesk_, seq_len_, epoch):
    """
    Set target length according to traing epoch,
    target len will increase along with epoch.

    target_outputs_: np.array
    seq_len_: dict
    epoch: int

    return:
    target_outputs_: np.array
    """

    target_steps = seq_len_['output_len']
    min_out_len = min([target_steps, 2])
    new_len = min([min_out_len + epoch, target_steps])
    delta = abs(target_steps - new_len)

    target_outputs_ = target_outputs_[:, :seq_len_['seq_len'] - delta, :]
    input_data_ = input_data_[:, :seq_len_['seq_len'] - delta, :]
    mesk_ = mesk_[:, :seq_len_['seq_len'] - delta, :]

    seq_len_['output_len'] = seq_len_['output_len'] - delta
    seq_len_['seq_len'] = seq_len_['seq_len'] - delta

    return input_data_, target_outputs_, seq_len_, mesk_
