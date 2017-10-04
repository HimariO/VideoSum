import warnings
import numpy as np
import pickle
import getopt
import time
import sys
import os
import csv
import spacy
import re
import math
import random

from PIL import Image
from termcolor import colored
from train_until import *


class VideoCaptionDataset:

    def __init__(self, root_dir, annotation, lexicon_dict, iter_cap=3000000, dataset_size=30000, show_log=False, batch_size=1):
        self.show_log = show_log
        self.root_dir = root_dir
        self.annotation = annotation
        self.lexicon_dict = lexicon_dict

        self.batch_size = batch_size
        self.dataset_size = dataset_size

        self.iter_cap = iter_cap
        self.current_iter = 0
        self.npy_info = self._get_npy_info()

        self.current_npy = (None, -1, -1, None)  # [npy, start_id, end_id, keys]
        self.current_batch = {
            'input_data': None,
            'target_outputs': None,
            'seq_len': None,
            'mask': None,
        }

    def lprint(self, message):
        if self.show_log:
            print(message)

    def _get_npy_info(self):
        feat_files = [re.match('features_(\d+)_(\d+)\.npy', f) for f in os.listdir(path=self.root_dir)]
        assert len(feat_files) > 0

        feat_files_tup = []
        for f in feat_files:
            if f is not None:
                feat_files_tup.append((f.string, int(f.group(1)), int(f.group(2))))  # (file_name, start_id, end_id)
        feat_files_tup.sort(key=lambda x: x[1])

        return feat_files_tup

    def _next_npy(self):
        data_ID = self.current_iter % self.dataset_size

        if self.current_npy is None or self.current_npy[1] > data_ID or self.current_npy[2] < data_ID:
            for npy in self.npy_info:
                if npy[1] <= data_ID and npy[2] >= data_ID:
                    self.lprint('Loading %s...' % npy[0])
                    npy_feats = np.load(os.path.join(self.root_dir, py[0])).tolist()
                    self.current_npy = npy_feats, npy[1], npy[2], list(npy_feats.keys())
                    break
                else:
                    self.current_npy = (None, -1, -1, None)

            if self.current_npy[1:3] == (-1, -1) and self.current_npy[0] is None:  # if no data are aliviable.
                raise ValueError('Can\'t find suitable data for tarining.')

            # if i >= reuse_data_param * data_size:  # update input seq len, if train on same data more than one time.
            #     reuse_data_param = math.ceil(i / data_size / 2)
            #     seq_video_num = 1 + ((reuse_data_param - 1) * 1) % 1

    def _seq_append(self, input_data_, target_outputs_, seq_len_, mask_):
        self.current_batch['input_data'] = np.concatenate([self.current_batch['input_data'], input_data_], axis=1) if self.current_batch['input_data'] is not None else input_data_
        self.current_batch['target_outputs'] = np.concatenate([self.current_batch['target_outputs'], target_outputs_], axis=1) if self.current_batch['target_outputs'] is not None else target_outputs_
        self.current_batch['seq_len'] = self.current_batch['seq_len'] + seq_len_['seq_len'] if self.current_batch['seq_len'] is not None else seq_len_['seq_len']
        self.current_batch['mask'] = np.concatenate([self.current_batch['mask'], mask_], axis=1) if self.current_batch['mask'] is not None else mask_
        # target_step = np.concatenate([target_step, target_step_], axis=0) if target_step is not None else target_step_

    def get_batchs(self):

        for I in range(self.iter_cap):
            self.current_iter = I
            self._next_npy()

            anno = self.annotation[I % self.dataset_size]
            vid_name = get_video_name(anno)

            self.current_batch = {
                'input_data': None,
                'target_outputs': None,
                'seq_len': None,
                'mask': None,
            }
