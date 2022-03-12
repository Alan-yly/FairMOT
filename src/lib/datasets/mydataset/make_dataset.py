import os
import json
import tqdm
import shutil
import cv2
from os.path import join
from os import listdir
def make_mot17_half_dataset(src_pth,dist_pth,id_offset):
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

        f = open(join(src_pth, seq, 'gt', 'gt.txt'), 'r')
        mot_result[seq] = {}
        lines = f.readlines()
        f.close()
        for line in lines:
            chars = line.split(',')
            for i in range(len(chars)-1):
                chars[i] = int(float(chars[i]))
            chars[-1] = float(chars[-1])
            frame_id = chars[0]
            vis_ratio = chars[-1]
            if chars[6] != 1 or vis_ratio < threshold:
                continue
            if chars[0] not in mot_result[seq].keys():
                mot_result[seq][frame_id] = []
            tlbr = [chars[2],chars[3],chars[2]+chars[4],chars[3]+chars[5]]
            mot_result[seq][frame_id].append((chars[1],tlbr,vis_ratio))
    id_map = {}
    out_result = {}
    val_result = {}
    for seq in tqdm.tqdm(mot_seq):
        len_files = len(listdir(join(src_pth, seq, 'img1'))) // 2
        for frame in mot_result[seq].keys():
            if frame <= len_files:
                jpg = join(src_pth,seq,'img1','{0:06d}.jpg'.format(frame))
                out_result[jpg] = []
                for x in mot_result[seq][frame]:
                    id = '-'.join([seq , str(x[0])])
                    if id not in id_map.keys():
                        id_map[id] = len(id_map)
                    out_result[jpg].append((id_map[id]+id_offset,x[1],x[2]))
        for frame in mot_result[seq].keys():
            if frame > len_files:
                jpg = join(src_pth,seq,'img1','{0:06d}.jpg'.format(frame))
                val_result[jpg] = []
                for x in mot_result[seq][frame]:
                    id = '-'.join([seq , str(x[0])])
                    if id not in id_map.keys():
                        continue
                    val_result[jpg].append((id_map[id]+id_offset,x[1],x[2]))

    f = open(join(dist_pth,'mot17_train_half.json'),'w')
    json.dump(out_result,f)
    f.close()
    f = open(join(dist_pth, 'mot17_val_half.json'), 'w')
    json.dump(val_result, f)
    f.close()
    return len(id_map)

def make_PRW_CUHK_dataset(src_pth,dist_pth,id_offset):
    files = listdir(join(src_pth,'images'))
    out_result = {}
    id_map = {}
    labels = listdir(join(src_pth,'labels'))
    no_labels = 0
    for file in tqdm.tqdm(files):
        tmp = []
        if file[:-4]+'.txt' not in labels:
            no_labels+=1
            continue
        f = open(join(src_pth,'labels',file[:-4]+'.txt'),'r')
        lines = f.readlines()
        f.close()
        import cv2
        img = cv2.imread(join(src_pth, 'images', file))
        w = img.shape[1]
        h = img.shape[0]
        for line in lines:
            chars =  line.split()
            tlbr = chars[2:]
            for i in range(4):
                tlbr[i] = float(tlbr[i])
            tlbr[0] *= w
            tlbr[1] *= h
            tlbr[0] = tlbr[0] - w * tlbr[2] / 2
            tlbr[1] = tlbr[1] - h * tlbr[3] / 2
            tlbr[2] = tlbr[0] + w * tlbr[2]
            tlbr[3] = tlbr[1] + h * tlbr[3]
            for i in range(4):
                tlbr[i] = int(tlbr[i])
            # cv2.rectangle(img,(tlbr[0],tlbr[1]),(tlbr[2],tlbr[3]),(255,0,0),2)
            # cv2.imshow('img',img)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            for i in range(len(chars)):
                chars[i] = int(float(chars[i]))
            if chars[1] == -1:
                continue
            if chars[1] not in id_map.keys():
                id_map[chars[1]] = len(id_map)
            tmp.append((id_map[chars[1]]+id_offset,tlbr,1.))
        if len(tmp) != 0:
            out_result[join(src_pth, 'images', file)] = tmp
    f = open(dist_pth,'w')
    json.dump(out_result,f)
    f.close()
    print(no_labels)
    return len(id_map)
def merge_json(src_path,dist_pth):
    files = listdir(src_path)
    out = {}
    for file in files:
        if file[-4:] == 'json' and file != 'all_labels.json':
            f = open(join(src_path,file),'r')
            tmp = json.load(f)
            for key in tmp.keys():
                out[key] = tmp[key]
    f = open(join(dist_pth,'all_lables.json'),'w')
    json.dump(out,f)
    f.close()



if __name__ == '__main__':
    id_offset = 0
    id_offset += make_mot17_half_dataset('/home/hust/yly/Dataset/MOT17/train','/home/hust/yly/Dataset/',id_offset)
    print(id_offset)
    id_offset += make_PRW_CUHK_dataset('/home/hust/newdisk/PRW','/home/hust/yly/Dataset/PRW.json',id_offset)
    print(id_offset)
    id_offset += make_PRW_CUHK_dataset('/home/hust/newdisk/CUHK-SYSU', '/home/hust/yly/Dataset/CUHK.json', id_offset)
    print(id_offset)
    merge_json('/home/hust/yly/Dataset/','/home/hust/yly/Dataset/')

