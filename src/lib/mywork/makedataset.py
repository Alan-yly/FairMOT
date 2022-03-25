import os
import json
import tqdm
import shutil
import cv2
from os.path import join
from os import listdir
# from ..models.model import create_model,load_model
min_frames = 30

def make_mot17_half_dataset(src_pth,dist_pth):
    mot_seq = ["MOT17-02-SDP",
               'MOT17-04-SDP',
               'MOT17-05-SDP',
               'MOT17-09-SDP',
               'MOT17-10-SDP',
               'MOT17-11-SDP',
               'MOT17-13-SDP']
    mot_result = {}
    threshold = 0.2
    for seq in tqdm.tqdm(mot_seq):
        len_files = len(listdir(join(src_pth, seq, 'img1'))) // 2
        f = open(join(src_pth, seq, 'gt', 'gt.txt'), 'r')
        mot_result[seq] = {}
        lines = f.readlines()
        f.close()
        id_map = {}
        for line in lines:
            chars = line.split(',')
            for i in range(len(chars)-1):
                chars[i] = int(float(chars[i]))
            chars[-1] = float(chars[-1])
            track_id = chars[1]
            frame_id = chars[0]
            vis_ratio = chars[-1]
            if chars[6] != 1 or vis_ratio < threshold or frame_id > len_files:
                continue
            if track_id not in id_map.keys():
                id_map[track_id] = len(id_map)
                id = id_map[track_id]
                mot_result[seq][id] = [[] for _ in range(len_files)]
            id = id_map[track_id]
            tlbr = [chars[2],chars[3],chars[2]+chars[4],chars[3]+chars[5]]
            mot_result[seq][id][frame_id-1] = (join(src_pth,seq,'img1','{0:06d}.jpg'.format(frame_id)),tlbr,vis_ratio)
        id = 0
        for key in range(len(mot_result[seq])):
            valid_frames = 0
            for frame in mot_result[seq][key]:
                if len(frame) > 0:
                    valid_frames += 1
            tmp = mot_result[seq][key]
            del mot_result[seq][key]
            if valid_frames >= 2*min_frames:
                mot_result[seq][id] = tmp
                id += 1
    f = open(join(dist_pth,'mot17_half.json'),'w')
    json.dump(mot_result,f)
    f.close()
    return


def make_mot15_dataset(src_pth, dist_pth):
    mot_seq = ["TUD-Campus",
               'PETS09-S2L1',
               'KITTI-17',
               'KITTI-13',
               'ETH-Sunnyday',
               'ETH-Pedcross2',
               'ETH-Bahnhof',
               'ADL-Rundle-8',
               'ADL-Rundle-6',
               'Venice-2',
               'TUD-Stadtmitte']
    mot_result = {}
    for seq in tqdm.tqdm(mot_seq):
        len_files = len(listdir(join(src_pth, seq, 'img1')))
        f = open(join(src_pth, seq, 'gt', 'gt.txt'), 'r')
        mot_result[seq] = {}
        lines = f.readlines()
        f.close()
        id_map = {}
        for line in lines:
            chars = line.split(',')
            for i in range(len(chars)):
                chars[i] = int(float(chars[i]))
            track_id = chars[1]
            frame_id = chars[0]
            if chars[6] != 1:
                continue
            if track_id not in id_map.keys():
                id_map[track_id] = len(id_map)
                id = id_map[track_id]
                mot_result[seq][id] = [[] for _ in range(len_files)]
            id = id_map[track_id]
            tlbr = [chars[2], chars[3], chars[2] + chars[4], chars[3] + chars[5]]
            mot_result[seq][id][frame_id - 1] = (join(src_pth,seq,'img1','{0:06d}.jpg'.format(frame_id)),tlbr, 1.)
        id = 0
        for key in range(len(mot_result[seq])):
            valid_frames = 0
            for frame in mot_result[seq][key]:
                if len(frame) > 0:
                    valid_frames += 1
            tmp = mot_result[seq][key]
            del mot_result[seq][key]
            if valid_frames >= 2*min_frames:
                mot_result[seq][id] = tmp
                id += 1
    f = open(join(dist_pth, 'mot15.json'), 'w')
    json.dump(mot_result, f)
    f.close()
    return

def make_mot20_dataset(src_pth,dist_pth):
    mot_seq = ["MOT20-01",
               'MOT20-02',
               'MOT20-03',
               'MOT20-05',
            ]
    mot_result = {}
    threshold = 0.2
    for seq in tqdm.tqdm(mot_seq):
        len_files = len(listdir(join(src_pth, seq, 'img1')))
        f = open(join(src_pth, seq, 'gt', 'gt.txt'), 'r')
        mot_result[seq] = {}
        lines = f.readlines()
        f.close()
        id_map = {}
        for line in lines:
            chars = line.split(',')
            for i in range(len(chars)-1):
                chars[i] = int(float(chars[i]))
            chars[-1] = float(chars[-1])
            track_id = chars[1]
            frame_id = chars[0]
            vis_ratio = chars[-1]
            if chars[6] != 1 or vis_ratio < threshold:
                continue
            if track_id not in id_map.keys():
                id_map[track_id] = len(id_map)
                id = id_map[track_id]
                mot_result[seq][id] = [[] for _ in range(len_files)]
            id = id_map[track_id]
            tlbr = [chars[2],chars[3],chars[2]+chars[4],chars[3]+chars[5]]
            mot_result[seq][id][frame_id-1] = (join(src_pth,seq,'img1','{0:06d}.jpg'.format(frame_id)),tlbr,vis_ratio)
        id = 0
        for key in range(len(mot_result[seq])):
            valid_frames = 0
            for frame in mot_result[seq][key]:
                if len(frame) > 0:
                    valid_frames += 1
            tmp = mot_result[seq][key]
            del mot_result[seq][key]
            if valid_frames >= 2*min_frames:
                mot_result[seq][id] = tmp
                id += 1
    f = open(join(dist_pth,'mot20.json'),'w')
    json.dump(mot_result,f)
    f.close()
    return

def merge_result(src_pth):
    out = {}
    files = ['mot15.json','mot17_half.json']
    for file in listdir(src_pth):
        if file[-4:] == 'json' and file in files:
            ob = json.load(open(join(src_pth,file),'r'))
            for key in ob.keys():
                out[key] = ob[key]
    f = open(join(src_pth, 'all.json'), 'w')
    json.dump(out, f)
    f.close()


if __name__ == '__main__':
    make_mot17_half_dataset('/home/hust/yly/Dataset/MOT17/train','/home/hust/yly/Dataset/')
    make_mot15_dataset('/home/hust/yly/Dataset/MOT15/train', '/home/hust/yly/Dataset/')
    make_mot20_dataset('/home/hust/yly/Dataset/MOT20/train', '/home/hust/yly/Dataset/')
    merge_result('/home/hust/yly/Dataset/')

