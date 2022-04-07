import numpy as np
from cython_bbox import bbox_overlaps as bbox_ious
import os
data_root = '/home/hust/yly/Dataset/MOT16/train/'
seqs = os.listdir(data_root)
seqs = sorted(seqs)
for seq in seqs:
    path_in = os.path.join(data_root,seq,'gt/gt.txt')
    input_ = np.loadtxt(path_in, delimiter=',')
    input_ = input_[np.lexsort([input_[:, 1], input_[:, 0]])]  # 按ID和帧排序
    input_ = input_[input_[:,-3] == 1]
    input_[:,4] += input_[:,2]
    input_[:,5] += input_[:,3]

    max_frame = int(np.max(input_[:,0]))
    sum_ob = 0
    sum_iou = 0
    print(seq+':')
    thres = [i*0.2 + 0.1 for i in range(5)]
    for thre in thres:
        for i in range(1,max_frame+1):
            objects = input_[input_[:,0] == i]
            objects = objects[objects[:,-1] > thre]
            bbox = objects[:,2:6]
            iou_dist = bbox_ious(bbox,bbox)
            num_ob = len(bbox)
            sum_iou += np.sum(iou_dist > 0.2) - num_ob
            sum_ob += num_ob
        print(sum_iou / sum_ob)
