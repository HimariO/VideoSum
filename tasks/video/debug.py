import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import re
import os
from train_until import *
import spacy
import tensorflow as tf
from dnc.memory import KMemory

nlp = spacy.load('en')

anno_file = './dataset/MSR_en.csv'
dict_file = './dataset/MSR_en_dict.csv'
w2v_dict_file = './dataset/MSR_enW2V_dict.csv'
video_dir = './dataset/YouTubeClips/'
word2v_emb_file = './dataset/MSR_enW2V.npy'


def get_dataset(path='./dataset/'):
    feat_files = [re.match('features_(\d+)_(\d+)\.npy', f) for f in os.listdir(path=path)]
    feat_files_tup = []
    for f in feat_files:
        if f is not None:
            feat_files_tup.append((
                os.path.join(path, f.string),
                int(f.group(1)),
                int(f.group(2))
            ))  # (file_name, start_id, end_id)
    feat_files_tup.sort(key=lambda x: x[1])  # sort by start data id.
    return feat_files_tup


def plotVideo(feat):

    def fun(x, y, arr):
        if y < arr.shape[0] and x < arr.shape[1]:
            return arr[y, x]
        return 0

    feat = feat.tolist()
    # increase number of frame to 10 time, so we can see it better on plot.
    feat10 = []
    for f in feat:
        for _ in range(5):
            feat10.append(f)

    feat10 = np.array(feat10)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = np.arange(0, feat10.shape[1], 1)  # feat size
    y = np.arange(0, feat10.shape[0], 1)  # feat num
    X, Y = np.meshgrid(x, y)

    zs = np.array([fun(x, y, feat10) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z)
    # ax.plot_wireframe(X, Y, Z)

    ax.set_xlabel('FC6 Feature')
    ax.set_ylabel('Video Frame')
    ax.set_zlabel('Feature Value')

    # fig.imshow(feat10)
    plt.show()


def onehot_vec2id(nparr):
    result = []
    for batch in nparr:
        batch_vec = []
        for step in batch:
            batch_vec.append(step.argmax())
        result.append(batch_vec)
    return np.array(result)

data, lexicon_dict = load(anno_file, w2v_dict_file)
dataset = get_dataset()
d100 = np.load(dataset[0][0]).tolist()
k100 = list(d100.keys())

v_feat_means = []
feat_dic = {'some_word': ['CNN feature']}

for key in k100:
    v_feature = d100[key]
    if type(v_feature) is list:
        v_feature = np.array(v_feature)
    if len(v_feature.shape) < 2:
        continue
    v_feat_means.append(v_feature.mean(axis=0))

# for anno in data[10:40]:
#     output_str = [str(i) for i in nlp(anno['Description'][:-5])]
#     print(output_str)
#     input('waiting...')

v_feat_means = np.array(v_feat_means)

# n, bins, patches = plt.hist(v_feat_means[:, 123])
# plt.show()

# plotVideo(v_feat_means)

debug = np.load('debug_step-1001.npy')
debug = debug.tolist()

tar = debug['target']
out = debug['softmax_out']
raw_out = debug['raw_out']
grad = debug['grad']
mask = debug['mask']

# debug2 = np.load('debug_step-1270625.npy')
# debug2 = debug2.tolist()
#
# tar2 = debug2['target']
# out2 = debug2['softmax_out']
# raw_out2 = debug2['raw_out']
# grad2 = debug2['grad']
# mask2 = debug2['mask']

print(mask.shape)
with tf.Graph().as_default():
    with tf.Session() as sess:

        def loss_debug():
            tar_cut = tar[:, 7:12, :]
            raw_out_cut = raw_out[:, 7:12, :]

            tar_p = tf.placeholder(tf.float32, shape=tar.shape[1:])
            raw_out_p = tf.placeholder(tf.float32, shape=raw_out.shape[1:])

            b_tar_p = tf.placeholder(tf.int32, shape=tar.shape[:-1])
            b_raw_out_p = tf.placeholder(tf.float32, shape=raw_out.shape)
            mask_p = tf.placeholder(tf.float32, shape=mask.shape[:-1])

            tar_cut_p = tf.placeholder(tf.float32, shape=tar_cut.shape[1:])
            raw_out_cut_p = tf.placeholder(tf.float32, shape=raw_out_cut.shape[1:])

            t_tar_cut_p = tf.placeholder(tf.float32, shape=tar_cut.shape[2])
            t_raw_out_cut_p = tf.placeholder(tf.float32, shape=raw_out_cut.shape[2])

            loss = tf.nn.softmax_cross_entropy_with_logits(labels=tar_p, logits=raw_out_p)
            loss_cut = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tar_cut_p, logits=raw_out_cut_p))
            # loss_seq = tf.contrib.keras.backend.binary_crossentropy(raw_out_cut_p, tar_cut_p, from_logits=True)
            loss_seq = tf.reduce_sum(tf.contrib.keras.backend.binary_crossentropy(t_raw_out_cut_p, t_tar_cut_p, from_logits=True), axis=0)
            # loss_seq = tf.reduce_mean(loss_seq)

            A, B, C = sess.run([loss, loss_cut, loss_seq], feed_dict={
                raw_out_p: raw_out[0],
                raw_out_cut_p: raw_out_cut[0],

                tar_p: tar[0],
                tar_cut_p: tar_cut[0],

                t_tar_cut_p: tar[0, 4],
                t_raw_out_cut_p: tar_cut[0, 4],

                b_tar_p: onehot_vec2id(tar),
                b_raw_out_p: raw_out,
                mask_p: mask.reshape([1, 12])
            })

            print(A)
            print(B)
            print(C)
            print(C.shape)

        def KMem_debug():
            weight = np.random.normal(size=[1, 512])

            memory = KMemory(512, 512, 4, 1)
            weight_pt = tf.placeholder(tf.float32, shape=[1, 512])
            k_wt, debug_list = memory.K_weight(weight_pt, debug=True)

            result_wt, ind, sortlist = sess.run([k_wt] + debug_list, feed_dict={
                weight_pt: weight
            })

            print(weight)
            print(weight[0].max())
            print(weight[0].argsort()[-8:])

            print('-' * 100)

            print(result_wt)
            print(ind, '\n', sortlist)

        KMem_debug()
