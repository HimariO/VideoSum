import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def fun(x, y, arr):
    if y < arr.shape[0] and x < arr.shape[1]:
        return arr[y, x]
    return 0

feat = np.load('features.npy')
feat = feat

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.arange(0, feat.shape[1], 1)  # feat size
y = np.arange(0, feat.shape[0], 1)  # feat num
X, Y = np.meshgrid(x, y)

zs = np.array([fun(x, y, feat) for x, y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

ax.plot_surface(X, Y, Z)
# ax.plot_wireframe(X, Y, Z)

ax.set_xlabel('FC6 Feature')
ax.set_ylabel('Video Frame')
ax.set_zlabel('Feature Value')

plt.show()
