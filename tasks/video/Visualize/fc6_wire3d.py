import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def fun(x, y, arr):
    if y < arr.shape[0] and x < arr.shape[1]:
        return arr[y, x]
    return 0

# feat = np.load('features.npy')

def plotVideo2D(feat):
    feat10 = []
    for f in feat:
        for _ in range(5):
            feat10.append(f)
    plt.imshow(feat10)
    plt.show()

def plotVideo(feat):
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

    fig.imshow(feat10)
    plt.show()


def plotVideoAvg(feat):
    feat = feat.sum(axis=0) / feat.shape[0]

    fig = plt.figure()
    plt.bar(list(range(feat.shape[0])), feat)
    plt.show()
