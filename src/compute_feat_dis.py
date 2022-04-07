import numpy as np
from cython_bbox import bbox_overlaps as bbox_ious
import os
import lap
from lib.models.model import create_model,load_model
import torch
import  torch.nn.functional as F
import torchvision
import cv2
import tqdm
import random
model = create_model('dla_34', {'hm': 1, 'wh': 4, 'id': 128, 'reg': 2},256)
model = load_model(model, '/home/hust/yly/Model/fairmot_dla34.pth').cuda()


data_root = '/home/hust/yly/Dataset/MOT20/train/'
seqs = os.listdir(data_root)
seqs = sorted(seqs)

minvr = 0.8
maxvr = 1
hista = []
histb = []
for seq in seqs:
    path_in = os.path.join(data_root,seq,'gt/gt.txt')
    gt = np.loadtxt(path_in, delimiter=',')
    gt = gt[gt[:,-3] == 1]
    gt[:,4] += gt[:,2]
    gt[:,5] += gt[:,3]


    max_frame = int(np.max(gt[:,0]))
    print(seq+':')

    import collections
    id_feat = collections.defaultdict(list)

    for i in tqdm.tqdm(range(max_frame//2+1,max_frame+1)):
        objects = gt[gt[:,0] == i]
        objects = objects[minvr <= objects[:,-1] ]
        objects = objects[objects[:, -1] <= maxvr]
        if len(objects) == 0:
            continue
        tlbr = objects[:,2:6]
        img_pth = os.path.join(data_root,seq,'img1','{0:06d}.jpg'.format(i))
        with torch.no_grad():
            img = cv2.imread(img_pth)
            def letterbox(img, height=608, width=1088,
                          color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular
                shape = img.shape[:2]  # shape = [height, width]
                ratio = min(float(height) / shape[0], float(width) / shape[1])
                new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
                dw = (width - new_shape[0]) / 2  # width padding
                dh = (height - new_shape[1]) / 2  # height padding
                top, bottom = round(dh - 0.1), round(dh + 0.1)
                left, right = round(dw - 0.1), round(dw + 0.1)
                img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
                img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                         value=color)  # padded rectangular
                return img, ratio, dw, dh
            img, r, padw, padh = letterbox(img)

            tlbr = np.array(tlbr).astype(np.float)
            tlbr *= r
            tlbr[:,0] += padw
            tlbr[:,1] += padh
            tlbr[:,2] += padw
            tlbr[:,3] += padh

            tlbr = tlbr / 4
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img, dtype=np.float32)
            img /= 255.0
            img = torch.from_numpy(img).to('cuda').unsqueeze(0)
            output = model(img)[-1]
            id_feature = output['id']
            id_feature = F.normalize(id_feature, dim=1)
            tlbr = torch.from_numpy(tlbr).cuda()
            x = (tlbr[:,0] + tlbr[:,2]) /2
            x = x.unsqueeze(-1)
            y = (tlbr[:,1] + tlbr[:,3]) /2
            y = y.unsqueeze(-1)
            z = torch.zeros_like(x).cuda()
            z = torch.cat([z,x,y,x,y],dim=1).float()
            feat = torchvision.ops.roi_align(id_feature,z,1).squeeze()
        for i in range(len(objects)):
            id_feat[int(objects[i,1])].append(feat[i])

    def func(lista,listb,hist):
        for feata in random.sample(lista,min(10,len(lista))):
            for featb in random.sample(listb,min(10,len(listb))):
                n = torch.sum(feata*featb).cpu().numpy()
                if n > 1 or n < -1:
                    print('!'*100)
                hist.append(1-n)
    for keysa in tqdm.tqdm(id_feat.keys()):
        func(id_feat[keysa],id_feat[keysa],hista)
        # for ind,feata in enumerate(id_feat[keysa]):
        #     for featb in id_feat[keysa][max(0,ind-1):min(ind,len(id_feat[keysa]))]:
        #         n = torch.sum(feata*featb).cpu().numpy()
        #         if n > 1 or n < -1:
        #             print('!'*100)
        #         hista.append(1-n)
        keysb = keysa
        while keysb == keysa:
            keysb = random.sample(id_feat.keys(),1)[0]
        func(id_feat[keysa],id_feat[keysb],histb)

hista = np.array(hista)
histb = np.array(histb)
avga = np.mean(hista)
avgb = np.mean(histb)
print("%.4f" % avga + '±' + "%.4f" % np.sqrt(np.mean((hista-avga)**2)))
print("%.4f" % avgb + '±' + "%.4f" % np.sqrt(np.mean((histb - avgb) ** 2)))
import matplotlib.pyplot as plt
bins = np.linspace(0, 2, 100)
fig = plt.figure()
ax = fig.add_subplot(111)

ax.hist(histb, label='diff target',histtype='stepfilled', alpha=0.5, density=True, bins=bins)
ax.hist(hista, label='same target',histtype='stepfilled', alpha=0.5, density=True, bins=bins)

seq = 'visratio:' + '['+str(minvr) +','+ str(maxvr)+']'
# seq = seq[:4] + '7' + seq[5:]
ax.legend(loc='upper left',title=seq)

plt.show()


