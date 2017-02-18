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

video_dir = './dataset/YouTubeClips/'

datas = []

with open('./dataset/MSR_en.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        datas.append(row)

feats = []

with tf.Session() as sess:
    vgg = Vgg19(vgg19_npy_path='./VGG/vgg19.npy')
    image_holder = tf.placeholder('float', [1, 224, 224, 3])
    vgg.build(image_holder)

    for annotation, ind in zip(datas, range(len(datas))):

        try:
            path = video_dir + '%s_%s_%s.avi' % (annotation['VideoID'], annotation['Start'], annotation['End'])
            frames = load_video(path)

            feat = []
            for f in frames:
                f5_3 = sess.run([vgg.conv5_3], feed_dict={image_holder: f})
                feat.append(f5_3)

            print('frame count %d' % len(feat))
            feats.append(feat)
        except:
            print(colored('error', color='red'))
        break  # one time only

print('Start Saving...')
feats = np.array(feats)
np.save('.features.npy', feats)
