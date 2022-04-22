import numpy as np
from cython_bbox import bbox_overlaps as bbox_ious
import os
import lap
def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b

data_root = '/home/hust/yly/Dataset/MOT17/train/'
seqs = os.listdir(data_root)
seqs = sorted(seqs)
import collections
res_mat = [collections.defaultdict(int),collections.defaultdict(int)]
res_sum = [collections.defaultdict(int),collections.defaultdict(int)]
seqs_group = ['02','04','09']
# seqs_group = []
for seq in seqs:
    if seq[-3:] != 'DPM':
        continue
    path_in = os.path.join(data_root,seq,'gt/gt.txt')
    det_in = os.path.join(data_root,seq,'det/det.txt')
    gt = np.loadtxt(path_in, delimiter=',')
    gt = gt[np.lexsort([gt[:, 1], gt[:, 0]])]  # 按ID和帧排序
    gt = gt[gt[:,-3] == 1]
    gt[:,4] += gt[:,2]
    gt[:,5] += gt[:,3]

    det = np.loadtxt(det_in, delimiter=',')
    det = det[np.lexsort([det[:, 1], det[:, 0]])]  # 按ID和帧排序
    det[:, 4] += det[:, 2]
    det[:, 5] += det[:, 3]
    max_frame = int(np.max(gt[:,0]))
    print(seq+':')

    import collections

    sum_match = collections.defaultdict(int)
    sum_ob = collections.defaultdict(int)
    for i in range(1,max_frame+1):
        objects = gt[gt[:,0] == i]
        bbox = objects[:,2:6]
        for box in objects:
            sum_ob[int(box[-1]/0.2)] += 1

        dets = det[det[:,0] == i]
        dets = dets[dets[:, -4] > 0]
        dets = dets[:,2:6]


        iou_dist = 1-bbox_ious(bbox,dets).astype(np.float32)
        match,_,_ = linear_assignment(iou_dist,0.5)
        for m in match:
            if m[0] == -1 or m[1] == -1:
                continue
            sum_match[int(objects[m[0],-1] / 0.2)] += 1
    keys = []
    for key in sum_ob.keys():
        keys.append(key)
    keys = sorted(keys)
    for key in keys:
        print(sum_match[key] / sum_ob[key])
        if seq[6:8] in seqs_group:
            res_mat[0][key] += sum_match[key]
            res_sum[0][key] += sum_ob[key]
        else:
            res_mat[1][key] += sum_match[key]
            res_sum[1][key] += sum_ob[key]


print('no motion')
keys = []
for key in sum_ob.keys():
    keys.append(key)

keys = sorted(keys)
for key in keys:
    print('[{0},{1})'.format(key*0.2,key*0.2+0.2))
for key in res_mat[0].keys():
    print(res_mat[0][key] / res_sum[0][key])
print('move')
for key in res_mat[1].keys():
    print(res_mat[1][key] / res_sum[1][key])