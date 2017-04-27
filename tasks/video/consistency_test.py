import os
import re
import train_until
import random
from Visualize.CNNs_feat_distribution.get_single_videofeat import Extractor
from PIL import Image
import numpy as np

"""
This script check for preprocessed feature.npy file's id are matching data annotation.
"""

anno_file = './dataset/MSR_en.csv'
dict_file = './dataset/MSR_en_dict.csv'
w2v_dict_file = './dataset/MSR_enW2V_dict.csv'
video_dir = './dataset/YouTubeClips/'
word2v_emb_file = './dataset/MSR_enW2V.npy'


feat_files = [re.match('features_(\d+)_(\d+)\.npy', f) for f in os.listdir(path='./dataset/')]
feat_files_tup = []
for f in feat_files:
    if f is not None:
        feat_files_tup.append((f.string, int(f.group(1)), int(f.group(2))))  # (file_name, start_id, end_id)
feat_files_tup.sort(key=lambda x: x[1])  # sort by start data id.

data, _ = train_until.load(anno_file, w2v_dict_file)

test_data_id = [int(random.random() * 30000) for i in range(10)]
model = Extractor()

test_result = []

for ID in test_data_id:
    feat_npy = []
    file_tup = None

    for f in feat_files_tup:
        if ID >= f[1] and ID <= f[2]:
            feat_npy = np.load('./dataset/' + f[0])
            file_tup = f

    if file_tup is None:
        print('data [%d] not found in ./dataset.' % ID)
        continue

    D = data[ID]
    gen_feat = np.array(feat_npy[ID - file_tup[1]])

    video_path = '%s_%s_%s.avi' % (D['VideoID'], D['Start'], D['End'])
    if os.path.isfile(video_dir + video_path):
        frames = train_until.load_video(video_dir + video_path, use_VGG=False)
        print('Load video [%s] from ./dataset.' % video_path)
    else:
        print('video [%s] not found in ./dataset.' % video_path)
        continue

    for frame in frames:
        feat = model.extract_PIL(Image.fromarray(frame))

        if np.any(gen_feat[:] == feat):
            test_result.append(True)
            break
        else:
            print(np.any(gen_feat[:] == feat))

if all(test_result):
    print('[ Sucess! ]')
else:
    print('[ Fail! ]')
print(test_result)
