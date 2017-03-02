import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import getopt
import sys

options, _ = getopt.getopt(sys.argv[1:], '', ['file='])
filename = 'Ugb_uH72d0I_8_17_memView_27K.npy'
dpi = 150
fig_size = (2560 / dpi, 1440 / dpi)

for opt in options:
    if opt[0] == '--file':
        filename = opt[1]

mem_dict = np.load(filename)[()]

Ww = np.reshape(mem_dict['write_weightings'], mem_dict['write_weightings'].shape[1:])
Wr = np.reshape(mem_dict['read_weightings'], mem_dict['read_weightings'].shape[1:])
Vu = np.reshape(mem_dict['usage_vectors'], mem_dict['usage_vectors'].shape[1:])

fig = plt.figure(figsize=fig_size, dpi=dpi)
fig.suptitle(filename)

a = fig.add_subplot(2, 3, 1)
imgplot = plt.imshow(Ww)
a.set_title('write_weightings')
a.set_ylabel('step')
plt.colorbar(orientation='vertical')

a = fig.add_subplot(2, 3, 2)
imgplot = plt.imshow(Vu)
a.set_title('usage_vectors')
a.set_ylabel('step')
plt.colorbar(orientation='vertical')
# plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')

for i in range(4):
    a = fig.add_subplot(2, 3, 3 + i)
    imgplot = plt.imshow(Wr[:, :, i])
    # imgplot.set_clim(0.0, 0.7)
    a.set_title('readhead[%d]_weightings' % i)
    a.set_ylabel('step')
    plt.colorbar(orientation='vertical')

fig2 = plt.figure(figsize=fig_size, dpi=dpi)
fig2.suptitle(filename)

Ga = np.reshape(mem_dict['allocation_gates'], mem_dict['allocation_gates'].shape[:-1])
Gw = np.reshape(mem_dict['write_gates'], mem_dict['write_gates'].shape[:-1])
Gf = mem_dict['free_gates']

a = fig2.add_subplot(2, 3, 1)
imgplot = plt.imshow(Ga)
a.set_title('allocation_gates')
a.set_xlabel('step')
plt.colorbar(orientation='horizontal')

a = fig2.add_subplot(2, 3, 2)
imgplot = plt.imshow(Gw)
a.set_title('write_gates')
a.set_xlabel('step')
plt.colorbar(orientation='horizontal')

for i in range(4):
    a = fig2.add_subplot(2, 3, 3 + i)
    imgplot = plt.imshow(Gf[:, :, i])
    a.set_title('readhead[%d]_weightings' % i)
    a.set_xlabel('step')
    plt.colorbar(orientation='horizontal')

# plt.show()
plt.tight_layout()
fig.savefig(filename[:-4] + '_f1.jpg', dpi=dpi)
fig2.savefig(filename[:-4] + '_f2.jpg', dpi=dpi)
