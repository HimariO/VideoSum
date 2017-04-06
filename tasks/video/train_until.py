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

anno_file = './dataset/MSR_en.csv'
dict_file = './dataset/MSR_en_dict.csv'
w2v_dict_file = './dataset/MSR_enW2V_dict.csv'
video_dir = './dataset/YouTubeClips/'
word2v_emb_file = './dataset/MSR_enW2V.npy'
nlp = spacy.load('en')


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
                   word_emb=None, extend_target=False, setMask=True):
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

    output_str = [str(i) for i in nlp(annotation['Description'][:-5])] + [EOS]
    output_str = [dictionary[i] for i in output_str]
    if extend_target:
        temp = []
        for s in output_str:
            for _ in range(3):
                temp.append(s)
        output_str = temp

    print('video_seq: %d, target_seq: %d' % (input_vec.shape[0], len(output_str)))

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
        padding_i = np.zeros([seq_len - input_vec.shape[0], 2048], dtype=np.float32)
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


def prepare_mixSample(annotation, dictionary, video_feature, redu_sample_rate=1,
                      word_emb=None, extend_target=False, setMask=True, post_padding=0):
    """
    output input-output pair with overlay timestep.
    input_vector = [D, D, D, D, D, 0, 0, 0, 0]
    output_vector= [0, 0, 0, D, D, D, D, D, D]
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

    output_str = [str(i) for i in nlp(annotation['Description'][:-5])] + [EOS]
    output_str = [dictionary[i] for i in output_str]
    if extend_target:
        temp = []
        for s in output_str:
            for _ in range(3):
                temp.append(s)
        output_str = temp

    print('video_seq: %d, target_seq: %d' % (input_vec.shape[0], len(output_str)))

    over_lap = 3 if min([len(output_str), input_vec.shape[0]]) >= 3 else min([len(output_str), input_vec.shape[0]])

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
        padding_i = np.zeros([seq_len - input_vec.shape[0], 2048], dtype=np.float32)
        input_vec = np.concatenate([input_vec, padding_i], axis=0)

    if post_padding > 0:
        output_vec = np.concatenate([output_vec, np.zeros([post_padding, word_space_size])], axis=0)
        input_vec = np.concatenate([input_vec, np.zeros([post_padding, 2048])], axis=0)
        mask = np.concatenate([mask, np.zeros(post_padding)])
        seq_len += post_padding

    print(colored('seq_len: ', color='yellow'), seq_len)
    print(colored('input_vec: ', color='yellow'), input_vec.shape)
    print(colored('output_vec: ', color='yellow'), output_vec.shape)
    return (
        input_vec,
        np.reshape(output_vec, (1, -1, word_space_size)),
        seq_len,
        np.reshape(mask, (1, -1, 1))
    )
