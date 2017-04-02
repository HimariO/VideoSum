# That's an impressive list of imports.
import numpy as np
from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist

# We import sklearn.
import sklearn
from sklearn.manifold import TSNE
from sklearn.preprocessing import scale

# We'll hack a bit with the t-SNE code in sklearn 0.15.2.
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold.t_sne import (_joint_probabilities,
                                    _kl_divergence)
from sklearn.utils.extmath import _ravel

# We'll use matplotlib for graphics.
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# We import seaborn to make nice plots.
import seaborn as sns

import json
import os
from PIL import Image
import getopt
import sys

options, _ = getopt.getopt(sys.argv[1:], '', ['file='])
for opt in options:
    if opt[0] == '--file':
        feat_path = opt[1]


# image_paths = list(map(lambda img: './train2014/' + img['file_name'], anno['images']))
# image_ids = list(map(lambda img: img['id'], anno['images']))
sample_fold = './SampleVidImg'
image_paths = np.load('img_list.npy')


def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f = plt.figure()
    ax = plt.subplot()
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40,
                    c=palette[1])
    plt.xlim(-400, 400)
    plt.ylim(-400, 400)
    ax.axis('off')
    ax.axis('tight')

    img_boxs = []
    for ind, point in zip(range(len(x)), x):
        oImg = OffsetImage(plt.imread(image_paths[ind]), zoom=.2)
        ab = AnnotationBbox(oImg, xy=(point[0], point[1]), xycoords='data', boxcoords="offset points")
        img_boxs.append(ax.add_artist(oImg))
        print('ImgBox[%d]' % ind)

    return f, ax, sc, img_boxs


def scatter_PIL(p, size=(1000, 1000)):
    # create a white background.
    base = Image.new('RGB', size, color=1)

    x_max = max(p, key=lambda _p: _p[0])[0]
    y_max = max(p, key=lambda _p: _p[1])[1]

    x_min = min(p, key=lambda _p: _p[0])[0]
    y_min = min(p, key=lambda _p: _p[1])[1]

    # resize_scaler = max([x_max - x_min, y_max - y_min]) / size[0]
    resize_scaler = ((x_max - x_min) + (y_max - y_min)) / 2
    resize_scaler = (sum(size) / 2) / resize_scaler
    center_offset = ((x_max - x_min) / 2, (y_max - y_min) / 2)
    print(x_max, x_min, y_max, y_min)

    for i in range(len(p)):
        # p[i][0] -= center_offset[0]
        p[i][0] *= resize_scaler

        # p[i][1] -= center_offset[1]
        p[i][1] *= resize_scaler

    for ind, point in zip(range(len(p)), p):
        oImg = Image.open(image_paths[ind])
        _img = oImg.resize((int(oImg.size[0] * 0.5), int(oImg.size[1] * 0.5)))

        new_pos = (int(size[0] / 2 + point[0]), int(size[1] / 2 + point[1]))
        base.paste(_img, new_pos)

    return base

# Random state.
RS = 20150101

start_img = 1000
end_img = 2000

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

feats = np.load(feat_path)
feats_flat = []

for feat in feats:
    feats_flat.append(feat.reshape(-1))

print('Start TSNE...')
tsne_proj = TSNE(random_state=RS).fit_transform(feats_flat)
# print(tsne_proj)
print('Ploting...')
result_img = scatter_PIL(tsne_proj, size=(5000, 5000))
result_img.save(feat_path[:-4] + '_distrib.jpg')
# plt.savefig('./Plot/digits_tsne-generated.png', dpi=120)
