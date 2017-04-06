import numpy as np
from scipy.spatial import distance
from scipy.spatial import KDTree
from get_single_videofeat import Extractor, VGGExtractor
from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image
from termcolor import colored

import getopt
import sys


base_point = [
    './SampleVidImg/1200_[200_290].jpg',
    './SampleVidImg/1200_[320_380].jpg',
    './SampleVidImg/1200_[470_350].jpg',
    './SampleVidImg/1200_[620_290].jpg',
    './SampleVidImg/1200_[740_260].jpg',
    './SampleVidImg/1200_[920_320].jpg',
    './SampleVidImg/1260_[530_350].jpg',
    './SampleVidImg/1260_[680_260].jpg',
    './SampleVidImg/1260_[950_410].jpg',
    './SampleVidImg/1320_[140_350].jpg',
    './SampleVidImg/1320_[230_380].jpg',
    './SampleVidImg/1320_[830_200].jpg',
    './SampleVidImg/1380_[500_380].jpg',
    './SampleVidImg/1380_[620_230].jpg',
    './SampleVidImg/1440_[530_200].jpg',
]

img_list = np.load('img_list.npy')
incep3 = np.load('InceptionV3_feats.npy')
sort_ = np.load('mean_dif_IDsort.npy')

indexs = [np.where(img_list == base_point[i])[0][0] for i in range(len(base_point))]
base_vecs = []
for i in range(len(base_point)):
    ori_feat = incep3[indexs[i]]
    part_feat = [ori_feat[j] for j in sort_[-150:]]
    base_vecs.append(np.array(part_feat))

result_fold = './Picked'
options, _ = getopt.getopt(sys.argv[1:], '', ['file='])

for opt in options:
    if opt[0] == '--file':
        video_path = opt[1]

if __name__ == '__main__':
    clip = VideoFileClip(video_path, audio=False)

    coun = 0
    max_frame_cout = 100
    start_count = 60 * 100  # 60 fps * 40 sec

    imgs_path = []
    model = Extractor()

    for clip in clip.iter_frames():
        coun += 1

        if coun % 90 != 0 or coun < start_count:
            continue
        elif len(imgs_path) >= max_frame_cout:
            break

        img = Image.fromarray(clip)
        step = 30
        sample_size = (150, 200)
        margin = 80

        negitve_feat = []
        print(colored('getting: ', color='green'), len(imgs_path))

        for x in range(0 + margin, img.size[0] - sample_size[0] - margin, step):
            for y in range(0 + margin, img.size[1] - sample_size[1] - margin, step):
                crop = img.crop(
                    (x, y, x + sample_size[0], y + sample_size[1])
                )

                # section = lambda arr: np.concatenate([arr[:1500], arr[1499:1500]], axis=0)
                feat = model.extract_PIL(crop)
                part_feat = [feat[j] for j in sort_[-150:]]
                part_feat = np.array(part_feat)

                min_dice = min(
                    [np.linalg.norm(base_vecs[i] - part_feat) for i in range(len(base_point))]
                )
                print('%d_[%d_%d] ' % (coun, x, y), min_dice)

                if min_dice < 4.5:
                    crop.save(result_fold + '/%d_[%d_%d].jpg' % (coun, x, y))
                    imgs_path.append(result_fold + '/%d_[%d_%d].jpg' % (coun, x, y))  # for recording number of output
                # elif min_dice > 15:
                #     negitve_feat.append(feat)

    # np.save('false_sample.npy', np.array(negitve_feat))
    # np.save('postive_sample.npy', np.array([incep3[indexs[i]] for i in range(len(base_point))]))
