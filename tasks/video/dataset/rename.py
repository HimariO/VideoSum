import os
import re

feat_files = [re.match('features_(\d+)_(\d+)\\\.npy', f) for f in os.listdir(path='./')]
feat_files_tup = []
for f in feat_files:
    if f is not None:
        feat_files_tup.append((f.string, int(f.group(1)), int(f.group(2))))  # (file_name, start_id, end_id)
feat_files_tup.sort(key=lambda x: x[1])  # sort by start data id.

for f in feat_files_tup:
    os.rename(f[0], 'features_%d_%d.npy' % (f[1], f[2]))
