import json
import os

import cv2

import numpy as np
from cython_bbox import bbox_overlaps as bbox_ious
def tlwh_to_tlbr(tlwh):
    ret = np.asarray(tlwh).copy()
    ret[2:] += ret[:2]
    return ret

from cython_bbox import bbox_overlaps as bbox_ious
class losttrack_viser():
    def __init__(self, img_path,out_path,begin_frame):
        self.result = []
        self.img_path = img_path
        self.begin_frame = begin_frame
        self.f = open(out_path,'w')
    def __del__(self):
        json.dump(self.result,self.f)
        self.f.close()
    def append(self,frame_id,last_id,tlbr,last_tlbr):
        img1 = os.path.join(self.img_path,'{0:06d}.jpg'.format(self.begin_frame+frame_id-1))
        img2 = os.path.join(self.img_path,'{0:06d}.jpg'.format(self.begin_frame+last_id-1))
        self.result.append((img1,img2,tlbr.tolist(),last_tlbr.tolist()))


if __name__ == '__main__':
    import random
    gt = {}
    base_path = 'G:\project\MOT\DataSet\\train'
    files =  os.listdir(base_path)
    objects = []
    for file in files:
        if file[-4:] == 'json':
            seq = file[-23:-11]
            print(seq)
            gt[seq] = {}
            lines = open(os.path.join(base_path,seq,'gt/gt.txt'),'r')
            for line in lines:
                tmps = line.split(',')
                for i in range(8):
                    tmps[i] = int(tmps[i])
                if tmps[6] == 0:
                    continue
                if tmps[0] not in gt[seq].keys():
                    gt[seq][tmps[0]] = []
                gt[seq][tmps[0]].append((tlwh_to_tlbr(tmps[2:6]),tmps[1]))
            object = json.load(open(os.path.join(base_path,file),'r'))
            objects += object
    random.shuffle(objects)
    print(len(objects))
    err_ob = 0
    data = []
    pie = [0,0,0,0]
    for ob in objects:
        seq = ob[0][35:35+12]
        dets1 = gt[seq][int(ob[0][-10:-4])]
        dets2 = gt[seq][int(ob[1][-10:-4])]
        tlbrs = []
        for det in dets1:
            tlbrs.append(det[0])
        dist = 1-bbox_ious(np.array(tlbrs).astype(np.float),np.array([ob[2]]))
        if np.min(dist) >0.5:
            id1 = -1
        else:
            id1 = dets1[np.argmin(dist)][1]
            det1 = dets1[np.argmin(dist)][0]
            iou1 = np.min(dist)
        tlbrs = []
        for det in dets2:
            tlbrs.append(det[0])
        dist = 1 - bbox_ious(np.array(tlbrs).astype(np.float), np.array([ob[3]]))
        if np.min(dist) > 0.5:
            id2 = -2
        else:
            id2 = dets2[np.argmin(dist)][1]
            det2 = dets2[np.argmin(dist)][0]
            iou2 = np.min(dist)
        if id1 != id2:
            err_ob += 1
            frameid1 = int(ob[0][-10:-4])
            frameid2 = int(ob[1][-10:-4])
            # print(id1, id2, frameid1 - frameid2)
            data.append(frameid1 - frameid2)
            if id1 != -1 and id2 != -2:
                pie[0] += 1
            elif id1 == -1 and id2 != -2:
                pie[1] += 1
            elif id1 != -1 and id2 == -2:
                pie[2] += 1
            else:
                pie[3] += 1
            # if id2 != -2:
            #     print(iou1,frameid1)
            #     print(iou2,frameid2)
            #     print(id1,id2)
            #     img1 = cv2.imread(os.path.join(base_path,ob[0][35:]))
            #     img2 = cv2.imread(os.path.join(base_path,ob[1][35:]))
            #     cv2.rectangle(img1,(int(ob[2][0]),int(ob[2][1])),(int(ob[2][2]),int(ob[2][3])),(255,0,0),2)
            #     if id1 != -1:
            #         cv2.rectangle(img1, (int(det1[0]), int(det1[1])), (int(det1[2]), int(det1[3])), (0, 0, 255), 2)
            #     for det in dets1:
            #         if det[1] == id2:
            #             det = det[0]
            #             cv2.rectangle(img1, (int(det[0]), int(det[1])), (int(det[2]), int(det[3])), (0, 255, 0), 2)
            #     cv2.rectangle(img2, (int(ob[3][0]), int(ob[3][1])), (int(ob[3][2]), int(ob[3][3])), (255, 0, 0), 2)
            #     cv2.rectangle(img2, (int(det2[0]), int(det2[1])), (int(det2[2]), int(det2[3])), (0, 0, 255), 2)
            #     img = cv2.resize(cv2.vconcat([img1,img2]),None,None,0.5,0.5)
            #     cv2.imshow('img',img)
            #     cv2.waitKey()
            #     cv2.destroyAllWindows()
    import matplotlib.pyplot as plt
    data.sort()
    x = [0,0,0,0]
    for d in data:
        if d < 15:
            x[0] += 1
        elif d < 30:
            x[1] += 1
        elif d < 45:
            x[2] += 1
        else:
            x[3] += 1
    # plt.hist(data, bins=20, density=True,facecolor="blue", edgecolor="black")
    plt.pie(pie,None,['!!','=!','!=','=='],autopct='%0.2f%%')
    plt.show()
    print(err_ob)
    print(pie)
