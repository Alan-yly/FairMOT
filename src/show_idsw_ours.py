import cv2
import pandas as pd
import numpy as np
data_root = 'G:\project\mot17_half_yolov5s'
gt_root = 'G:\project\MOT\DataSet\\train'
import os
for file in os.listdir(data_root):
    if file[-3:] == 'txt':
        print(file)
        seq = file[:-4]
        track_file = np.loadtxt(os.path.join(data_root,file),delimiter=',')
        events = pd.read_csv(os.path.join(data_root,file[:-4] + '.csv'))
        gt = np.loadtxt(os.path.join(gt_root,seq,'gt/gt.txt'),delimiter=',')
        max_frame = np.max(gt[:,0])
        start_frame = max_frame // 2 + 1
        idsws = events[events['Type'] == 'SWITCH']
        for _,idsw in idsws.iterrows():
            frame_id = int(idsw['FrameId'])
            def func(data,id,frame_id):
                data = data[data[:,0] == frame_id]

                data = data[data[:,1] == id]
                if len(data) == 0:
                    return []
                tlwh = data[0, 2: 6]
                tlbr = tlwh
                tlbr[2:] += tlbr[:2]
                tlbr = tlbr.tolist()
                for i in range(4):
                    tlbr[i] = int(tlbr[i])
                return tlbr
            hid = idsw['HId']
            oid = idsw['OId']
            track_tlbr = func(track_file,idsw['HId'],frame_id+start_frame)
            gt_tlbr = func(gt,idsw['OId'],frame_id+start_frame)
            last_track_tlbr = func(track_file,idsw['HId'],frame_id+start_frame-1)
            if len(last_track_tlbr) == 0:
                continue
            tracks = events[events['HId'] == idsw['HId']]
            last_frame_track =  tracks[tracks['FrameId'] == frame_id-1]
            for _,x in last_frame_track[last_frame_track['Type'] == 'MATCH'].iterrows():
                last_oid = x['OId']
            last_gt_tlbr = func(gt,last_oid,frame_id+start_frame-1)
            if len(last_gt_tlbr) == 0:
                continue
            print(hid,oid,last_oid)
            print(os.path.join(data_root,'img1','{0:06d}.jpg'.format(int(frame_id+start_frame))))
            img1 = cv2.imread(os.path.join(gt_root,seq,'img1','{0:06d}.jpg'.format(int(frame_id+start_frame))))
            img2 = cv2.imread(os.path.join(gt_root,seq, 'img1', '{0:06d}.jpg'.format(int(frame_id + start_frame-1))))
            cv2.rectangle(img1,(track_tlbr[0],track_tlbr[1]),(track_tlbr[2],track_tlbr[3]),(255,0,0),2)
            cv2.rectangle(img2, (last_track_tlbr[0], last_track_tlbr[1]), (last_track_tlbr[2], last_track_tlbr[3]), (255, 0, 0), 2)

            cv2.rectangle(img1, (gt_tlbr[0], gt_tlbr[1]), (gt_tlbr[2], gt_tlbr[3]), (0, 255, 0), 2)
            cv2.rectangle(img2, (last_gt_tlbr[0], last_gt_tlbr[1]), (last_gt_tlbr[2], last_gt_tlbr[3]),
                          (0, 255, 0), 2)
            img1 = cv2.resize(img1,None,None,0.5,0.5)
            img2 = cv2.resize(img2, None, None, 0.5, 0.5)
            cv2.imshow('img1',img1)
            cv2.imshow('img2',img2)
            cv2.waitKey()
            print('!')






