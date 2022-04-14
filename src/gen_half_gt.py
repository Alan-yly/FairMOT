import numpy as np

import os

import torchvision

import tqdm



#
data_root = '/home/hust/yly/TrackEval/data/gt/mot_challenge/MOT17-train'
seqs = os.listdir(data_root)
seqs = sorted(seqs)
#
for seq in seqs:

    path_in = os.path.join(data_root,seq,'gt/gt.txt')
    print(path_in)
    gt = np.loadtxt(path_in, delimiter=',')
    max_frame = int(np.max(gt[:,0]))

    gt = gt[gt[:,0]>= (max_frame//2+1)]
    np.savetxt(path_in,gt,fmt='%d,%d,%d,%d,%d,%d,%d,%d,%f,')

