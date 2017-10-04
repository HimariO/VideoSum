import numpy as np
import os
import tensorflow as tf

from VGG.vgg19 import Vgg19
import VGG.utils as utils
import math
from termcolor import colored
from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image
import csv
from Visualize.CNNs_feat_distribution.get_single_videofeat import Extractor


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


def get_video_feat(path):
    feat = []
    if use_VGG:
        frames = load_video(path)
        for f in frames:
            f5_3 = sess.run([vgg.fc6], feed_dict={image_holder: f})
            feat.append(f5_3)
    else:
        clip = VideoFileClip(path)
        skip = 6
        count = 0
        for f in clip.iter_frames():
            count += 1
            if count % skip != 0:
                continue
            feat.append(model.extract_PIL(Image.fromarray(f)))
    return feat

video_dir = './dataset/YouTubeClips/'

datas = []

with open('./dataset/MSR_en.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        datas.append(row)

feats = {}

start = 33400
end = 60001
end = end if end < len(datas) else len(datas)
datas = datas[start:end]

num_per_file = 100
use_VGG = False

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    with tf.device('/gpu:1'):
        if use_VGG:
            vgg = Vgg19(vgg19_npy_path='./VGG/vgg19.npy')
            image_holder = tf.placeholder('float', [1, 224, 224, 3])
            vgg.build(image_holder)
        else:
            model = Extractor()
            # model = Extractor(layer="mixed10")

        for annotation, ind in zip(datas, range(len(datas))):
            source_data = '%s_%s_%s' % (annotation['VideoID'], annotation['Start'], annotation['End'])
            if os.path.isfile(video_dir + source_data + '.avi'):
                path = video_dir + source_data + '.avi'

                feat = get_video_feat(path)

                print('%d / %d' % (start + ind, end))
                feats[source_data] = feat
                print(annotation)
            else:
                # feats.append([0])
                feats[source_data] = [0]
                print(colored('error', color='red'))
            # break  # one time only
            if ((ind + 1) % num_per_file == 0 and ind != 0) or ind == len(datas) - 1:
                feats = np.array(feats)
                print(colored('Start Saving...', color='green'))
                np.save('dataset_output/features_%d_%d.npy' % (start + ind - num_per_file + 1, start + ind), feats)
                feats = {}
