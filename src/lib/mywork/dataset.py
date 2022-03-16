import glob
import math
import os
import os.path as osp
import random
import time
from collections import OrderedDict

import cv2
import json
import numpy as np
import torch
from ..models.model import create_model,load_model
import  torch.nn.functional as F
import torchvision


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


class JointDataset():  # for training
    def __init__(self, config,label_path, img_size=(1088, 608)):
        # Namespace(K=500, arch='dla_34', batch_size=12, cat_spec_wh=False, chunk_sizes=[6, 6], conf_thres=0.4,
        #           data_cfg='../src/lib/cfg/data.json', data_dir='/home/hust/yly/Dataset/', dataset='jde',
        #           debug_dir='/tmp/pycharm_project_382/src/lib/../../exp/mot/default/debug', dense_wh=False,
        #           det_thres=0.3, down_ratio=4, exp_dir='/tmp/pycharm_project_382/src/lib/../../exp/mot',
        #           exp_id='default', fix_res=True, gpus=[2, 3], gpus_str='2, 3', head_conv=256,
        #           heads={'hm': 1, 'wh': 4, 'id': 128, 'reg': 2}, hide_data_time=False, hm_weight=1, id_loss='ce',
        #           id_weight=1, img_size=(1088, 608), input_h=608, input_res=1088, input_video='../videos/MOT16-03.mp4',
        #           input_w=1088, keep_res=False, load_model='/home/hust/yly/Model/mix_mot17_half_dla34.pth', lr=0.0001,
        #           lr_step=[20], ltrb=True, master_batch_size=6, mean=[0.408, 0.447, 0.47], metric='loss',
        #           min_box_area=100, mse_loss=False, multi_loss='uncertainty', nID=14455, nms_thres=0.4, norm_wh=False,
        #           not_cuda_benchmark=False, not_prefetch_test=False, not_reg_offset=False, num_classes=1, num_epochs=30,
        #           num_iters=-1, num_stacks=1, num_workers=8, off_weight=1, output_format='video', output_h=152,
        #           output_res=272, output_root='../demos', output_w=272, pad=31, print_iter=0, reg_loss='l1',
        #           reg_offset=True, reid_dim=128, resume=False, root_dir='/tmp/pycharm_project_382/src/lib/../..',
        #           save_all=False, save_dir='/tmp/pycharm_project_382/src/lib/../../exp/mot/default', seed=317,
        #           std=[0.289, 0.274, 0.278], task='mot', test=False, test_hie=False, test_mot15=False, test_mot16=False,
        #           test_mot17=False, test_mot20=False, track_buffer=30, trainval=False, val_hie=False, val_intervals=5,
        #           val_mot15=False, val_mot16=False, val_mot17='True', val_mot20=False, vis_thresh=0.5, wh_weight=0.1)

        self.model = create_model('dla_34', {'hm': 1, 'wh': 4, 'id': 128, 'reg': 2},256)
        self.model = load_model(self.model, '/home/hust/yly/Model/mix_mot17_half_dla34.pth')
        self.model = self.model.to('cuda')
        self.model.eval()
        self.width = img_size[0]
        self.height = img_size[1]
        self.sum = []
        offset = 0
        self.label_files = json.load(open(label_path,'r'))
        for seq in self.label_files.keys():
            self.sum.append((offset,seq))
            offset += len(self.label_files[seq])
        self.nF = offset
        self.down_rate = config['down_rate']
        self.max_len = config['max_len']
        self.dim = 128

    def __getitem__(self, files_index):
        for ind in range(len(self.sum)):
            if files_index < self.sum[ind-1][0]:
                seq = self.sum[ind-1][1]
                files_index -= self.sum[ind-1][0]
                break



        track = self.label_files[seq][files_index]
        frames = []
        target_feat = torch.zeros(self.max_len,self.dim)
        ptrack_feat = torch.zeros(self.max_len, self.dim)
        ntrack_feat = torch.zeros(self.max_len, self.dim)
        for i in range(len(track)):
            if len(track[i]) != 0:
                frames.append(track[i])
        tmp = random.sample(range(len(frames)),2).sort()
        for ind,i in enumerate(range(tmp[0],min(len(frames),min(tmp[1],tmp[0]+self.max_len)))):
            target_feat[ind] = self.get_target_feat(frames[i][0],frames[i][1])
        for ind,i in enumerate(range(tmp[1],min(len(frames),tmp[1]+self.max_len))):
            ptrack_feat[ind] = self.get_target_feat(frames[i][0], frames[i][1])




        tmp = files_index
        while tmp == files_index:
            tmp = random.sample(range(len(self.label_files[seq])),1)
        ntrack = self.label_files[seq][tmp]
        frames = []
        for i in range(len(ntrack)):
            if len(ntrack[i]) != 0:
                frames.append(ntrack[i])
        tmp = random.sample(range(len(frames)),1).sort()
        for ind,i in enumerate(range(tmp[0], min(len(frames), tmp[0]+self.max_len))):
            ntrack_feat[ind] = self.get_target_feat(frames[i][0],frames[i][1])


        return target_feat,ptrack_feat,ntrack_feat
    def __len__(self):
        return self.nF
    def get_target_feat(self,img_pth,tlbr):
        with torch.no_grad():
            img = cv2.imread(img_pth)
            img, r, padw, padh = letterbox(img, height=self.height, width=self.width)
            tlbr = np.array(tlbr).astype(np.float)
            tlbr *= r
            tlbr[0] += padw
            tlbr[1] += padh
            tlbr[2] += padw
            tlbr[3] += padh
            tlbr = tlbr / self.down_rate
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img, dtype=np.float32)
            img /= 255.0
            img = torch.from_numpy(img).cuda().unsqueeze(0)
            output = self.model(img)[-1]
            id_feature = output['id']
            id_feature = F.normalize(id_feature, dim=1)
            x = (tlbr[0] + tlbr[2]) /2
            y = (tlbr[1] + tlbr[3]) /2
            return torchvision.ops.roi_align(id_feature,torch.tensor([0,x,y,x,y]).cuda(),1).squeeze()

