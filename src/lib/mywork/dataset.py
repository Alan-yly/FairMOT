import glob
import math
import os
import os.path as osp
import random
import time
from collections import OrderedDict
import json
import cv2
import json
import numpy as np
import torch
from ..models.model import create_model,load_model
import  torch.nn.functional as F
import torchvision


class Dataset():  # for training
    def __init__(self, config, img_size=(1088, 608)):
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
        self.device = config['device']
        self.model = create_model('dla_34', {'hm': 1, 'wh': 4, 'id': 128, 'reg': 2},256)
        self.model = load_model(self.model, config['feat_extract_model_path'])
        self.model = self.model.to(self.device)
        self.model.eval()
        self.width = img_size[0]
        self.height = img_size[1]
        self.sum = []
        offset = 0
        self.label_files = json.load(open(config['label_pth'],'r'))
        for seq in self.label_files.keys():
            offset += len(self.label_files[seq])
            self.sum.append((offset,seq))

        self.nF = offset
        self.down_rate = config['down_rate']
        self.max_len = config['max_len']
        self.dim = 128
        self.file = config['file']
        self.json = open(self.file,'r')
        self.recoder = {}
        self.recoder = json.load(self.json)
        self.json.close()
        self.min_frames = config['min_frames']
        self.max_frames = config['max_frames']


    def __del__(self):
        self.json = open(self.file, 'w')
        json.dump(self.recoder,self.json)
        print('finish record')
        self.json.close()
    def __getitem__(self, files_index):
        offset = 0
        for ind in range(0,len(self.sum)):
            if files_index < self.sum[ind][0]:
                seq = self.sum[ind][1]
                files_index -= offset
                break
            offset = self.sum[ind][0]

        track = self.label_files[seq][str(files_index)]
        frames = []
        target_feat = torch.zeros(self.max_len,self.dim).to(self.device)
        ptrack_feat = torch.zeros(self.max_len, self.dim).to(self.device)
        ntrack_feat = torch.zeros(self.max_len, self.dim).to(self.device)
        target_mask = torch.zeros(self.max_len).to(self.device)
        ptrack_mask = torch.zeros(self.max_len).to(self.device)
        ntrack_mask = torch.zeros(self.max_len).to(self.device)
        for i in range(len(track)):
            if len(track[i]) != 0:
                frames.append(track[i])

        tstart = random.sample(range(0,len(frames)-2*self.min_frames+1),1)[0]
        tend = random.sample(range(tstart+self.min_frames-1,min(len(frames)-self.min_frames,tstart+self.max_frames)),1)[0]
        pstart = random.sample(range(tend+1,len(frames)-self.min_frames+1),1)[0]
        pend = random.sample(range(pstart+1,min(len(frames),pstart+self.max_frames)),1)[0]
        for ind,i in enumerate(range(tstart,tend+1)):
            target_feat[ind] = self.get_target_feat(frames[i][0],frames[i][1],files_index)
            target_mask[ind] = 1
        for ind,i in enumerate(range(pstart,pend+1)):
            ptrack_feat[ind] = self.get_target_feat(frames[i][0], frames[i][1],files_index)
            ptrack_mask[ind] = 1



        idtmp = files_index
        while idtmp == files_index:
            idtmp = random.sample(range(len(self.label_files[seq])),1)[0]

        ntrack = self.label_files[seq][str(idtmp)]
        frames = []
        for i in range(len(ntrack)):
            if len(ntrack[i]) != 0:
                frames.append(ntrack[i])
        tmp = random.sample(range(len(frames)-self.min_frames+1),1)
        for ind,i in enumerate(range(tmp[0], min(len(frames), tmp[0]+random.randint(self.min_frames,self.max_frames)))):
            ntrack_feat[ind] = self.get_target_feat(frames[i][0],frames[i][1],idtmp)
            ntrack_mask[ind] = 1


        return (target_feat,target_mask),(ptrack_feat,ptrack_mask),(ntrack_feat,ntrack_mask)
    def __len__(self):
        return self.nF
    def get_target_feat(self,img_pth,tlbr,id):
        id = img_pth + '-'+str(id)
        if id in self.recoder.keys():
            return torch.from_numpy(np.array(self.recoder[id])).float().to(self.device)
        with torch.no_grad():
            img = cv2.imread(img_pth)
            # w = tlbr[2] - tlbr[0]
            # h = tlbr[3] - tlbr[1]
            # tlbr[0] += (random.random() - 0.5) * 0.1 * w
            # tlbr[1] += (random.random() - 0.5) * 0.1 * h
            # tlbr[2] += (random.random() - 0.5) * 0.1 * w
            # tlbr[3] += (random.random() - 0.5) * 0.1 * h
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
            tlbr[0] += padw
            tlbr[1] += padh
            tlbr[2] += padw
            tlbr[3] += padh

            tlbr = tlbr / self.down_rate
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img, dtype=np.float32)
            img /= 255.0
            img = torch.from_numpy(img).to(self.device).unsqueeze(0)
            output = self.model(img)[-1]
            id_feature = output['id']
            id_feature = F.normalize(id_feature, dim=1)
            x = (tlbr[0] + tlbr[2]) /2
            y = (tlbr[1] + tlbr[3]) /2
            feat = torchvision.ops.roi_align(id_feature,torch.tensor([[0,x,y,x,y]]).float().to(self.device),1).squeeze()
            self.recoder[id] = np.array(feat.cpu()).tolist()
            return feat
    # def get_target_feat(self,img_pth,tlbr,id):
    #     with torch.no_grad():
    #         img = cv2.imread(img_pth)
    #         img = img[:, :, ::-1].transpose(2, 0, 1)
    #         img = np.ascontiguousarray(img, dtype=np.float32)
    #         img = torch.from_numpy(img).to(self.device).unsqueeze(0)
    #         feat = torchvision.ops.roi_align(img,torch.tensor([[0,tlbr[0],tlbr[1],tlbr[2],tlbr[3]]]).float().cuda(),(16,8)).mean(1).reshape(128)
    #         return feat
