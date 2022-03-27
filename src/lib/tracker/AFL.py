"""
@Author: Du Yunhao
@Filename: AFLink.py
@Contact: dyh_bupt@163.com
@Time: 2021/12/28 19:55
@Discription: Appearance-Free Post Link
"""
import os
import glob
import torch
import numpy as np
from os.path import join, exists
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from ..mywork.mynetwork import Mynetwork
from ..models.model import create_model, load_model
import json
INFINITY = 1e5

class AFLink:
    def __init__(self, path_in, path_out,thres,thrT:tuple):
        self.thrP = thres
        self.thrT =thrT
        config = {'src_vocab':128,'trg_vocab':128,'d_model':128,'N':6,'heads':8,'dropout':0.2}
        self.model = Mynetwork(config)        # 预测模型
        self.path_out = path_out  # 结果保存路径
        self.track = np.loadtxt(path_in, delimiter=',')
        self.model.cuda()
        self.model.eval()
        self.feat =json.load(open('feat.json'))
    # 获取轨迹信息
    def gather_info(self):
        id2info = defaultdict(list)
        self.track = self.track[np.argsort(self.track[:, 0])]  # 按帧排序
        for row in self.track:
            f, i, x, y, w, h = row[:6]
            f = int(f)
            i = int(i)
            feat = np.zeros(128)
            if str(f) in self.feat[str(i)].keys():
                feat = np.array(self.feat[str(i)][str(f)])
            id2info[i].append([f,feat,x+w/2,y+h/2])
        self.track = np.array(self.track)
        id2info = {k: np.array(v) for k, v in id2info.items()}
        return id2info

    # 损失矩阵压缩
    def compression(self, cost_matrix, ids):
        # 行压缩
        mask_row = cost_matrix.min(axis=1) < self.thrP
        matrix = cost_matrix[mask_row, :]
        ids_row = ids[mask_row]
        # 列压缩
        mask_col = cost_matrix.min(axis=0) < self.thrP
        matrix = matrix[:, mask_col]
        ids_col = ids[mask_col]
        # 矩阵压缩
        return matrix, ids_row, ids_col

    # 连接损失预测
    def predict(self, track1, track2):
        max_len = 30
        feat1 = torch.zeros(1,max_len,128).cuda()
        mask1 = torch.ones(1,max_len).cuda()
        feat2 = torch.zeros(1,max_len,128).cuda()
        mask2 = torch.ones(1,max_len).cuda()
        for i,feat in enumerate(track1[-max_len:]):
            feat1[0,i] = torch.from_numpy(feat[1]).cuda()
            if (feat[1] == np.zeros(128)).all():
                mask1[0,i] = 0
        for i,feat in enumerate(track2[:max_len]):
            feat2[0,i] = torch.from_numpy(feat[1]).cuda()
            if (feat[1] == np.zeros(128)).all():
                mask2[0,i] = 0
        with torch.no_grad():
            score = self.model(feat1,feat2,mask1,mask2).squeeze().cpu().item()
        return score

    # 去重复: 即去除同一帧同一ID多个框的情况
    @staticmethod
    def deduplicate(tracks):
        _, index = np.unique(tracks[:, :2], return_index=True, axis=0)  # 保证帧号和ID号的唯一性
        return tracks[index]

    # 主函数
    def link(self):
        id2info = self.gather_info()
        num = len(id2info)  # 目标数量
        ids = np.array(list(id2info))  # 目标ID
        cost_matrix = np.ones((num, num)) * INFINITY  # 损失矩阵
        def f(x,y):
            import math
            return math.sqrt(x**2+y**2)
        '''计算损失矩阵'''
        for i, id_i in enumerate(ids):      # 前一轨迹
            for j, id_j in enumerate(ids):  # 后一轨迹
                if id_i == id_j: continue   # 禁止自娱自乐
                info_i, info_j = id2info[id_i], id2info[id_j]
                fi,xyi = info_i[-1][0] , info_i[-1][2:]
                fj,xyj = info_j[0][0], info_j[0][2:]
                if not self.thrT[0] <= fj - fi < self.thrT[1]: continue
                if f(xyi[0] - xyj[0] , xyi[1] - xyj[1]) > 75: continue
                cost = self.predict(info_i, info_j)
                if cost <= self.thrP: cost_matrix[i, j] = cost
        '''二分图最优匹配'''
        id2id = dict()  # 存储临时匹配结果
        ID2ID = dict()  # 存储最终匹配结果
        cost_matrix, ids_row, ids_col = self.compression(cost_matrix, ids)
        indices = linear_sum_assignment(cost_matrix)
        for i, j in zip(indices[0], indices[1]):
            if cost_matrix[i, j] < self.thrP:
                id2id[ids_row[i]] = ids_col[j]
        for k, v in id2id.items():
            if k in ID2ID:
                ID2ID[v] = ID2ID[k]
            else:
                ID2ID[v] = k
        # print('  ', ID2ID.items())
        '''结果存储'''
        res = self.track.copy()
        for k, v in ID2ID.items():
            res[res[:, 1] == k, 1] = v
        res = self.deduplicate(res)
        np.savetxt(self.path_out, res, fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d')



