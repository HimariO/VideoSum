from keras.preprocessing import image

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input as vgg_preprocess_input

from keras.models import Model, load_model
from keras.layers import Input
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image

import getopt
import sys

sample_fold = './SampleVidImg'
options, _ = getopt.getopt(sys.argv[1:], '', ['file='])
for opt in options:
    if opt[0] == '--file':
        video_path = opt[1]


class Extractor():
    def __init__(self, weights=None):
        """Either load pretrained from imagenet, or load our saved
        weights from our own training."""

        self.weights = weights  # so we can check elsewhere which model

        if weights is None:
            # Get model with pretrained weights.
            input_tensor = Input(shape=(299, 299, 3))
            base_model = InceptionV3(
                input_shape=(299, 299, 3),
                weights='imagenet',
                include_top=True
            )

            # We'll extract features at the final pool layer.
            self.model = Model(
                input=base_model.input,
                output=base_model.get_layer('avg_pool').output
            )

        else:
            # Load the model first.
            self.model = load_model(weights)

            # Then remove the top so we get features not predictions.
            # From: https://github.com/fchollet/keras/issues/2371
            self.model.layers.pop()
            self.model.layers.pop()  # two pops to get to pool layer
            self.model.outputs = [self.model.layers[-1].output]
            self.model.output_layers = [self.model.layers[-1]]
            self.model.layers[-1].outbound_nodes = []

    def extract(self, image_path):
        img = image.load_img(image_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Get the prediction.
        features = self.model.predict(x)
        print(image_path)

        if self.weights is None:
            # For imagenet/default network:
            features = features[0]
        else:
            # For loaded network:
            features = features[0]

        return features

    def extract_PIL(self, image):
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Get the prediction.
        features = self.model.predict(x)
        print(image_path)

        if self.weights is None:
            # For imagenet/default network:
            features = features[0]
        else:
            # For loaded network:
            features = features[0]

        return features


class VGGExtractor():
    def __init__(self, weights=None):
        """Either load pretrained from imagenet, or load our saved
        weights from our own training."""

        self.weights = weights  # so we can check elsewhere which model

        if weights is None:
            # Get model with pretrained weights.
            input_tensor = Input(shape=(224, 224, 3))
            base_model = VGG19(weights='imagenet', include_top=True)

            # We'll extract features at the final pool layer.
            self.model = Model(
                input=base_model.input,
                output=base_model.layers[-3].output
            )

        else:
            # Load the model first.
            self.model = load_model(weights)

            # Then remove the top so we get features not predictions.
            # From: https://github.com/fchollet/keras/issues/2371
            self.model.layers.pop()
            self.model.layers.pop()
            self.model.layers.pop()  # two pops to get to pool layer
            self.model.outputs = [self.model.layers[-1].output]
            self.model.output_layers = [self.model.layers[-1]]
            self.model.layers[-1].outbound_nodes = []

    def extract(self, image_path):
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = vgg_preprocess_input(x)

        # Get the prediction.
        features = self.model.predict(x)
        print(image_path)
        if self.weights is None:
            # For imagenet/default network:
            features = features[0]
        else:
            # For loaded network:
            features = features[0]

        return features

    def extract_PIL(self, image):
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Get the prediction.
        features = self.model.predict(x)
        print(features.shape)

        if self.weights is None:
            # For imagenet/default network:
            features = features[0]
        else:
            # For loaded network:
            features = features[0]

        return features


if __name__ == '__main__':
    clip = VideoFileClip(video_path, audio=False)

    coun = 0
    max_frame_cout = 2000
    start_count = 60 * 20  # 60 fps * 17 sec
    imgs_path = []

    for clip in clip.iter_frames():
        coun += 1

        if coun % 60 != 0 or coun < start_count:
            continue
        elif len(imgs_path) >= max_frame_cout:
            break

        img = Image.fromarray(clip)
        step = 30
        sample_size = (250, 250)
        margin = 80

        for x in range(0 + margin, img.size[0] - sample_size[0] - margin, step):
            for y in range(0 + margin, img.size[1] - sample_size[1] - margin, step):
                crop = img.crop(
                    (x, y, x + sample_size[0], y + sample_size[1])
                )
                crop.save(sample_fold + '/%d_[%d_%d].jpg' % (coun, x, y))
                imgs_path.append(sample_fold + '/%d_[%d_%d].jpg' % (coun, x, y))
        # img.save(sample_fold + '/%d.jpg' % coun)
        # imgs_path.append(sample_fold + '/%d.jpg' % coun)

    model = Extractor()
    feats = []

    for img_p in imgs_path:
        feats.append(model.extract(img_p))
    feats = np.array(feats)
    np.save('InceptionV3_feats.npy', feats)

    model = VGGExtractor()
    feats = []

    for img_p in imgs_path:
        feats.append(model.extract(img_p))
    feats = np.array(feats)
    np.save('VGG_feats.npy', feats)

    np.save('img_list.npy', imgs_path)
