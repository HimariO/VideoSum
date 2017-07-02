import numpy as np

debug = np.load('debug_target.npy')
debug = debug.tolist()

tar = debug['tar'][0]
out = debug['out'][0]

tar_state = []
out_state = []

# for i in range(tar.shape[0]):
#     tar_state.append(list(map(lambda x: x.any(), tar[i])))
#     out_state.append(list(map(lambda x: x.any(), out[i])))

tar_state = list(map(lambda x: x.any(), tar))
out_state = list(map(lambda x: x.any(), out))

tar_index = []
out_index = []

out_index = list(map(lambda x: x.argmax(), out))
tar_index = list(map(lambda x: x.argmax(), tar))

# for s, c in zip(out_state, range(len(out_state))):
#     if s is True:
#         out_index.append(c)
#         break
#
# for s, c in zip(tar_state, range(len(tar_state))):
#     if s is True:
#         tar_index.append(c)
#         break
