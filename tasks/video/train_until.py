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

from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image
from termcolor import colored
from sklearn import preprocessing


anno_file = './dataset/MSR_en.csv'
dict_file = './dataset/MSR_en_dict.csv'
w2v_dict_file = './dataset/MSR_enW2V_dict.csv'
video_dir = './dataset/YouTubeClips/'
word2v_emb_file = './dataset/MSR_enW2V.npy'
nlp = spacy.load('en')


def get_video_name(annotation):
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


def onehot(index, size):
    vec = np.zeros(size, dtype=np.float32)
    index = index if index < size else size - 1
    vec[index] = 1.0
    return vec


def prepare_sample(annotation, dictionary, video_feature, redu_sample_rate=1,
                   word_emb=None, extend_target=False, setMask=True, concat_input=False, norm=False):
    file_path = video_dir + '%s_%s_%s.avi' % (annotation['VideoID'], annotation['Start'], annotation['End'])
    EOS = '<EOS>' if word_emb is None else 'EOS'

    # choice output_szie depend on embedding method. (ont-hot or word2vec)
    if word_emb is None:
        word_space_size = len(dictionary)
    else:
        word_space_size = word_emb.shape[1]

    # some video in annotation does not include in dataset.
    if not os.path.isfile(file_path):
        raise OSError(2, 'No such file or directory', file_path)
    elif type(video_feature) is not np.ndarray:
        if video_feature == 0 or video_feature == [0]:
            raise OSError(2, 'Empty data slot!', file_path)

    video_feature = [np.reshape(i, [-1]) for i in video_feature]
    input_vec = np.array(video_feature[::redu_sample_rate])

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

    if word_emb is None:
        output_vec += [onehot(i, word_space_size) for i in output_str]
    else:
        output_vec += [word_emb[i] for i in output_str]

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
        # preprocessing.normalize(input_vec, axis=1)
        preprocessing.scale(input_vec, axis=1)

    print(colored('seq_len: ', color='yellow'), seq_len)
    # print(colored('input_vec: ', color='yellow'), input_vec.shape)
    # print(colored('output_vec: ', color='yellow'), output_vec.shape)
    return (
        np.reshape(input_vec, (1, -1, input_vec.shape[1])),
        np.reshape(output_vec, (1, -1, word_space_size)),
        {'seq_len': seq_len, 'input_len': input_len, 'output_len': output_len},
        np.reshape(mask, (1, -1, 1))
    )


def prepare_mixSample(annotation, dictionary, video_feature, redu_sample_rate=1,
                      word_emb=None, extend_target=False, setMask=True, post_padding=0, concat_input=False, norm=False):
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

    if word_emb is None:
        word_space_size = len(dictionary)
    else:
        word_space_size = word_emb.shape[1]

    # some video in annotation does not include in dataset.
    if not os.path.isfile(file_path):
        raise OSError(2, 'No such file or directory', file_path)
    elif type(video_feature) is not np.ndarray:
        if video_feature == 0 or video_feature == [0]:
            raise OSError(2, 'Empty data slot!', file_path)

    video_feature = [np.reshape(i, [-1]) for i in video_feature]
    input_vec = np.array(video_feature[::redu_sample_rate])

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
    if word_emb is None:
        output_vec = output_vec + [onehot(i, word_space_size) for i in output_str]
    else:
        output_vec = output_vec + [word_emb[i] for i in output_str]

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
    print(colored('seq_len: ', color='yellow'), seq_len)
    # print(colored('input_vec: ', color='yellow'), input_vec.shape)
    # print(colored('output_vec: ', color='yellow'), output_vec.shape)
    return (
        np.reshape(input_vec, (1, -1, input_vec.shape[1])),
        np.reshape(output_vec, (1, -1, word_space_size)),
        {'seq_len': seq_len, 'input_len': input_len, 'output_len': output_len},
        np.reshape(mask, (1, -1, 1))
    )


def prepare_batch(annotations, dictionary, video_features, epoch, redu_sample_rate=1,
                  word_emb=None, extend_target=False, setMask=True, useMix=False, accuTrain=False):
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
    none_padded = []

    for video_feature, annotation in zip(video_features, annotations):
        try:
            if useMix:
                input_data_, target_outputs_, seq_len_, mask_ = prepare_mixSample(annotation, dictionary, video_feature, redu_sample_rate, word_emb, extend_target, setMask)
            else:
                input_data_, target_outputs_, seq_len_, mask_ = prepare_sample(annotation, dictionary, video_feature, redu_sample_rate, word_emb, extend_target, setMask)

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
