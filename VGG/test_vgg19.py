import numpy as np
import tensorflow as tf

import vgg19
import utils

img1 = utils.load_image("./test_data/tiger.jpeg")
img2 = utils.load_image("./test_data/puzzle.jpeg")
img3 = utils.load_image("./test_data/large_10699.jpg")

batch1 = img1.reshape((1, 224, 224, 3))
batch2 = img2.reshape((1, 224, 224, 3))
batch3 = img3.reshape((1, 224, 224, 3))

batch = np.concatenate((batch1, batch2, batch3), 0)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

# with tf.Session(
#         config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
conv = None  # catch nparray of conv_layer feature.

with tf.Session() as sess:
    images = tf.placeholder("float", [3, 224, 224, 3])
    feed_dict = {images: batch}

    vgg = vgg19.Vgg19()
    with tf.name_scope("content_vgg"):
        vgg.build(images)

    prob, conv = sess.run([vgg.prob, vgg.conv5_3], feed_dict=feed_dict)
    print(prob)
    utils.print_prob(prob[0], './synset.txt')
    utils.print_prob(prob[1], './synset.txt')
