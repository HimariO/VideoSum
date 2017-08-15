import os
import sys
import json
import subprocess as SP
import time
import random
from termcolor import colored
import datetime

dataset_info = None

with open('./videodatainfo_2017.json', mode='r') as J_file:
    dataset_info = json.loads(J_file.read())

vid_info = dataset_info['videos']

for v in vid_info:
    print('Round: ', colored(v['id'], color='green'))

    if os.path.exists('./%s.mp4' % v['video_id']):
        continue

    args = [
        'youtube-dl',
        '-g',
        v['url'],
    ]

    dl_process = SP.Popen(args, stdout=SP.PIPE, stderr=SP.PIPE)
    out, err = dl_process.communicate()

    out = out.decode()
    out = out.split('\n')
    while True:
        try:
            out.remove('')
        except:
            break
    # print(out)
    for stream_url in out:
        args = [
            'ffmpeg',
            '-i',
            stream_url,
            '-ss',
            '00:%02d:%05.2f' % (v['start time'] // 60, v['start time'] % 60),
            '-to',
            '00:%02d:%05.2f' % (v['end time'] // 60, v['end time'] % 60),
            '-acodec',
            'libfdk_aac',
            '-c',
            'copy',
            '%s.mp4' % v['video_id'],
            '-y',
        ]

        ff_process = SP.Popen(args, stdout=SP.PIPE, stderr=SP.PIPE)
        out, err = ff_process.communicate()

        if ff_process.returncode == 0:
            break
        else:
            print(colored(args, color='blue'))
            print(out)
            print(err)

    print('SLeep', colored(datetime.datetime.now(), color='green'))
    time.sleep(2 * random.random())
