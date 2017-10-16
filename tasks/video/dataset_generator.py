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

    def __init__(self, root_dir, annotation, lexicon_dict, iter_cap=3000000, dataset_size=30000, show_log=False, batch_size=1, croups=None):
        self.show_log = show_log
        self.root_dir = root_dir
        self.annotation = annotation
        self.annotation_croups = croups
        self.lexicon_dict = lexicon_dict

        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.training_set_size = int(dataset_size * 0.9)
        self.val_set_size = dataset_size - self.training_set_size

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
        data_ID = self.current_iter % self.training_set_size

        if self.current_npy is None or self.current_npy[1] > data_ID or self.current_npy[2] < data_ID:
            for npy in self.npy_info:
                if npy[1] <= data_ID and npy[2] >= data_ID:
                    self.lprint('Loading %s...' % npy[0])
                    npy_feats = np.load(os.path.join(self.root_dir, npy[0])).tolist()
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

    def _get_single(self, I):
        self.current_iter = I
        self._next_npy()
        mixtrue_rate = 3

        # pair one normal video captions data(MSR dataset) with one text only data(bbc news dataset).
        if (self.annotation_croups is not None and I % mixtrue_rate != 0) or self.annotation_croups is None:
            anno = self.annotation[I % self.training_set_size]
            vid_name = get_video_name(anno)
            video_feat = self.current_npy[0][vid_name]
        else:
            anno = self.annotation_croups[(I // mixtrue_rate) % len(self.annotation_croups)]
            video_feat = np.zeros([1, 2048])  # TODO: hard code is BADDDD

        return anno, video_feat

    def get_batchs(self, start_iter=0, seq_concat=1, redu_sample_rate=3, feedback=False, word_emb=None, hamming=None, mul_onehot=None, norm=False, bucket=False):
        self.current_batch = {
            'input_data': None,
            'target_outputs': None,
            'seq_len': None,
            'mask': None,
            'num_in_batch': 0,
            'num_in_seq': 0,
        }

        raw_datas = {
            'anno': [],
            'video_feat': [],
        }

        for I in range(start_iter, self.iter_cap):
            anno, v_feat = self._get_single(I)
            try:
                raw_datas['anno'].append(anno)
                raw_datas['video_feat'].append(v_feat)

                self.current_batch['num_in_batch'] += 1

                if self.batch_size > self.current_batch['num_in_batch']:
                    # continue appending data into "raw_datas"
                    continue

                if self.batch_size == 1:
                    input_data_, target_outputs_, seq_len_, mask_ = prepare_sample(
                        raw_datas['anno'][0],
                        self.lexicon_dict,
                        raw_datas['video_feat'][0],
                        redu_sample_rate=redu_sample_rate,
                        word_emb=word_emb,
                        hamming=hamming,
                        mul_onehot=mul_onehot,
                        norm=norm,
                        bucket=bucket
                    )
                else:
                    input_data_, target_outputs_, seq_len_, mask_ = prepare_batch(
                        raw_datas['anno'],
                        self.lexicon_dict,
                        raw_datas['video_feat'],
                        redu_sample_rate=redu_sample_rate,
                        word_emb=word_emb,
                        hamming=hamming,
                        mul_onehot=mul_onehot,
                        norm=norm,
                        bucket=bucket
                    )

                if seq_len_['seq_len'] > 1200:
                    raise OSError('Too Long~')

                if feedback:
                    feed_b = np.roll(target_outputs_, 1, 1)
                    feed_b[:, 0, :] = 0
                    input_data_ = np.concatenate([input_data_, feed_b], axis=2)

                self.current_batch['num_in_seq'] += 1

                raw_datas = {
                    'anno': [],
                    'video_feat': [],
                }

                if self.current_batch['num_in_seq'] == 1:  # then data in "current_batch" are None.
                    self.current_batch.update({
                        'input_data': input_data_,
                        'target_outputs': target_outputs_,
                        'seq_len': seq_len_,
                        'mask': mask_,
                    })

                if seq_concat > self.current_batch['num_in_seq']:
                    if self.current_batch['num_in_seq'] > 1:  # there is already data is batch, just append to it.
                        self._seq_append(input_data_, target_outputs_, seq_len_, mask_)
                    continue

                self.lprint('[epoch] %d [iteration] %d/%d' % (I // self.training_set_size, I, self.iter_cap))
                yield self.current_batch

                self.current_batch = {
                    'input_data': None,
                    'target_outputs': None,
                    'seq_len': None,
                    'mask': None,
                    'num_in_batch': 0,
                    'num_in_seq': 0,
                }

            except OSError as err:
                del raw_datas['anno'][-1]
                del raw_datas['video_feat'][-1]
                self.lprint(err)

            except KeyError:
                del raw_datas['anno'][-1]
                del raw_datas['video_feat'][-1]
                self.lprint(colored('Error: ', color='red'), 'Annotation %s containing some word not exist in dictionary!' % sample['VideoID'])

    def get_val_batch(self):
        raise NotImplemented('Nope')
