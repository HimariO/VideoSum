import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation
import getopt
import sys

options, _ = getopt.getopt(sys.argv[1:], '', ['file='])
filename = 'Ugb_uH72d0I_8_17_memMatrix_step-81382.npy'
dpi = 150
fig_size = (2560 / dpi, 1440 / dpi)

for opt in options:
    if opt[0] == '--file':
        filename = opt[1]

mem_dict = np.load(filename).tolist()
# already get those from MemView.py
mem_dict.pop('usage_vector')
mem_dict.pop('read_weightings')
mem_dict.pop('write_weighting')

for k in mem_dict.keys():
    mem_dict[k] = mem_dict[k][0]

fig = plt.figure()
fig2 = plt.figure()
# fig3 = plt.figure()

link_mat = mem_dict['link_matrix']
mem_mat = mem_dict['memory_matrix']
read_vecs = mem_dict['read_vectors']

ims = []
ax1 = fig.add_subplot(1, 2, 1)
for i in range(link_mat.shape[0]):
    im = ax1.imshow(link_mat[i], animated=True)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=75, blit=False, repeat_delay=0)
# plt.colorbar(orientation='vertical')
# ani.save('dynamic_images.mp4', extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])

ims = []
ax2 = fig.add_subplot(1, 2, 2)
for i in range(mem_mat.shape[0]):
    im = ax2.imshow(mem_mat[i], animated=True)
    ims.append([im])

ani2 = animation.ArtistAnimation(fig, ims, interval=75, blit=False, repeat_delay=0)
# plt.colorbar(orientation='vertical')
# ani2.save('dynamic_images.mp4')
# plt.show()


# pos = [1, 2, 3, 4]
plot_num = read_vecs.shape[-1]

for i in range(read_vecs.shape[-1]):
    a = fig2.add_subplot(plot_num, 1, i+1)
    imgplot = plt.imshow(read_vecs[:, :, i])
    # imgplot.set_clim(0.0, 0.7)
    a.set_title('readhead[%d] read vector(values)' % i)
    a.set_ylabel('step')
    plt.colorbar(orientation='vertical')

plt.show()
