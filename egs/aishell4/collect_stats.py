import numpy as np
import kaldi_io

input_scp='/home/ouzj02/ankeyu/CAT-refactor/egs/aishell_conformer_singlech/data/train_sp/feats.scp'
output_file = 'stats.npy'
  
with open(input_scp) as f:
    lines = f.readlines()

s = 0
s2 = 0
count=0

for i, line in enumerate(lines):
    utt, feature_path = line.split()
    feature = kaldi_io.read_mat(feature_path)
    s += feature.sum(0)
    s2 += (feature**2).sum(0)
    count += np.shape(feature)[0]

s = np.pad(s, [0, 1], mode="constant", constant_values=count)
s2 = np.pad(s2, [0, 1], mode="constant", constant_values=0.0)
stats = np.stack([s, s2])
np.save(output_file, stats)    

