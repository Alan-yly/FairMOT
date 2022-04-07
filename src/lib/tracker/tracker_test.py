import numpy as np
from collections import deque
import itertools
import os
import os.path as osp
import time
import torch
import cv2
import torch.nn.functional as F
from ..models.model import create_model, load_model

from models.decode import mot_decode
from tracking_utils.utils import *
from tracking_utils.log import logger
from tracking_utils.kalman_filter import KalmanFilter
from models import *
from tracker import matching
from tracker import det_feat_record
from tracker import MAA
from .basetrack import BaseTrack, TrackState
from utils.post_process import ctdet_post_process
from utils.image import get_affine_transform
from models.utils import _tranpose_and_gather_feat
from tracker import det_feat_record
from tracker import refined_track_vis
from ..mywork.mynetwork import Mynetwork
from cython_bbox import bbox_overlaps as bbox_ious
from  collections import defaultdict

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, temp_feat, buffer_size,frame_id):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.score_list = []
        self.tracklet_len = 0

        self.smooth_feat = None
        self.update_features(temp_feat)
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.9

        self.sum_track_len = 0

        self.begin_frame_id = frame_id
        self.begin_tlbr = STrack.tlwh_to_tlbr(self._tlwh)
    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            # for i, st in enumerate(stracks):
            #     if st.state != TrackState.Tracked:
            #         multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.sum_track_len += 1
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        #self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.score_list.append(self.score)

        self.last_track_frame_id = frame_id
        self.last_track_tlbr = self.tlbr



    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )

        self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.sum_track_len += 1
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.score_list.append(self.score)

    def update(self, new_track, frame_id, update_feature=True):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """

        self.frame_id = frame_id
        self.tracklet_len += 1
        self.sum_track_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.score_list.append(self.score)
        if update_feature:
            self.update_features(new_track.curr_feat)

        self.last_track_frame_id = frame_id
        self.last_track_tlbr = self.tlbr
        # self.update_theta()
    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)

class JDETracker(object):
    def __init__(self, opt, frame_rate=30):
        self.opt = opt
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')
        print('Creating model...')
        self.model = create_model(opt.arch, opt.heads, opt.head_conv)
        self.model = load_model(self.model, opt.load_model)
        self.model = self.model.to(opt.device)
        self.model.eval()

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        #self.det_thresh = opt.conf_thres
        self.det_thresh = opt.conf_thres
        self.buffer_size = 20
        self.max_time_lost = 30
        self.max_per_image = opt.K
        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)

        self.kalman_filter = KalmanFilter()

        """CMA Module compent"""
        self.last_detect = None
        self.map_matrix = np.matrix(np.eye(3))
        self.matrixs = []
        self.window_matrix = 0
        self.last_img = None
        self.window_shake_ratios = []
        """Insert Module compent"""
        self.max_num_insframe = 150
        '''recorder'''
        self.recorder = None
        self.viser = None

        '''transformer'''
        config = {'src_vocab':128,'trg_vocab':128,'d_model':128,'N':6,'heads':8,'dropout':0.2}

        self.id_map = {}
        self.init_tracks = {}
        self.min_frames = 10
        self.max_frames = 20
        self.len_rematch = 0

        '''AFL'''
        self.feat_record = defaultdict(dict)
        self.start_frame_id = None

        self.use_mat = False

        '''iou_mean'''
        self.iou_mean = 0
        self.match_num = 0
        self.lost_det = 0

        self.iou_dist_time = defaultdict(list)
    def post_process(self, dets, meta):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.opt.num_classes)
        for j in range(1, self.opt.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.opt.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)

        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.opt.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.opt.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def update(self, im_blob, img0):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        width = img0.shape[1]
        height = img0.shape[0]
        inp_height = im_blob.shape[2]
        inp_width = im_blob.shape[3]
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.opt.down_ratio,
                'out_width': inp_width // self.opt.down_ratio}

        ''' Step 1: Network forward, get detections & embeddings'''
        if self.recorder.mode == 'record':
            with torch.no_grad():
                output = self.model(im_blob)[-1]
                hm = output['hm'].sigmoid_()
                wh = output['wh']
                id_feature = output['id']
                id_feature = F.normalize(id_feature, dim=1)

                reg = output['reg'] if self.opt.reg_offset else None
                dets, inds = mot_decode(hm, wh, reg=reg, ltrb=self.opt.ltrb, K=self.opt.K)

                id_feature = _tranpose_and_gather_feat(id_feature, inds)
                id_feature = id_feature.squeeze()
                id_feature = id_feature.cpu().numpy()

            dets = self.post_process(dets, meta)
            dets = self.merge_outputs([dets])[1]

            remain_inds = dets[:, 4] > self.opt.conf_thres
            inds_low = dets[:, 4] > 0.2
            inds_high = dets[:, 4] < self.opt.conf_thres
            inds_second = np.logical_and(inds_low, inds_high)

            dets_second = dets[inds_second]
            id_feature_second = id_feature[inds_second]
            dets = dets[remain_inds]
            id_feature = id_feature[remain_inds]
            '''record dets feats'''
            self.recorder.record((dets.tolist(),id_feature.tolist()),(dets_second.tolist(),id_feature_second.tolist()))
            '''record dets feats'''
        elif self.recorder.mode == 'get':
            a,b = self.recorder.get()
            dets = np.array(a[0])
            id_feature = np.array(a[1])
            dets_second = np.array(b[0])
            id_feature_second = np.array(b[1])

        """compute the map matrix"""
        rescale_ratio = 0.25
        img0 = cv2.resize(img0, None, None, rescale_ratio, rescale_ratio)
        self.get_two_img_map_matrix(img0, rescale_ratio)
        inv_map_matrix = np.linalg.inv(
            np.linalg.inv(self.matrixs[max(-self.window_matrix - 1, -len(self.matrixs))]) * self.map_matrix)
        for i in range(len(dets)):
            tlbr = self.compute_mapped_tlbr(dets[i, :4], inv_map_matrix)
            dets[i,:4] = tlbr
        for i in range(len(dets_second)):
            tlbr = self.compute_mapped_tlbr(dets_second[i, :4], inv_map_matrix)
            dets_second[i,:4] = tlbr
        """compute the map matrix"""


        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, self.buffer_size,self.frame_id) for
                          (tlbrs, f) in zip(dets[:, :5], id_feature)]
        else:
            detections = []

        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, self.buffer_size,self.frame_id) for
                                 (tlbrs, f) in zip(dets_second[:, :5], id_feature_second)]
        else:
            detections_second = []




        # vis
        '''
        for i in range(0, dets.shape[0]):
            bbox = dets[i][0:4]
            cv2.rectangle(img0, (bbox[0], bbox[1]),
                          (bbox[2], bbox[3]),
                          (0, 255, 0), 2)
        cv2.imshow('dets', img0)
        cv2.waitKey(0)
        id0 = id0-1
        '''



        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)


        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)



        ''' Step 2: First association, with embedding'''
        r_tracked_stracks,detections = self.match(strack_pool,detections,'embedding',0.4,activated_starcks,refind_stracks)

        for track in r_tracked_stracks[::-1]:
            if track.state == TrackState.Lost:
                r_tracked_stracks.pop(r_tracked_stracks.index(track))
        ''' Step 4: Third association, with IOU'''
        second_tracked_stracks,detections = self.match(r_tracked_stracks,detections,'Iou',0.5,activated_starcks,refind_stracks)

        ''' Step 5: association whit IOU on low score detection'''
        # second_tracked_stracks, _ = self.match(second_tracked_stracks, detections_second, 'Iou', 0.4, activated_starcks,
        #                                        refind_stracks)


        for track in second_tracked_stracks:
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        # detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        #self.tracked_stracks = remove_fp_stracks(self.tracked_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        logger.debug('===========Frame {}=========='.format(self.frame_id))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))


        """compute the map matrix"""
        out = []
        for strack in output_stracks:
            tlbr = self.compute_mapped_tlbr(strack.tlbr, np.linalg.inv(
                self.matrixs[max(-self.window_matrix - 1, -len(self.matrixs))]) * self.map_matrix)
            tlbr[2:] = tlbr[2:] - tlbr[:2]
            out.append((tlbr, strack.track_id))
        """compute the map matrix"""
        insert_frame = {}
        insert_frame = self.insert_frame_lost_track(refind_stracks)

        '''record tracked_track feature'''
        for track in output_stracks:
            self.feat_record[track.track_id][self.frame_id+self.start_frame_id] = track.curr_feat.tolist()
        return out,{}

    def compute_mapped_tlbr(self,tlbr,mat):
        tl = np.matrix(np.hstack([tlbr[:2], np.ones(1)]))
        tl = tl * mat
        tl = np.squeeze(np.array(tl))[:-1]
        tlbr[:2] = tl
        br = np.matrix(np.hstack([tlbr[2:], np.ones(1)]))
        br = br * mat
        br = np.squeeze(np.array(br))[:-1]
        tlbr[2:] = br
        return tlbr
    def compute_mapped_track(self,strack,mat):
        mean_xyah = strack.mean.copy()
        xy = np.matrix(np.hstack([mean_xyah[:2], np.ones(1)]))
        xy = xy * mat
        xy = np.squeeze(np.array(xy))[:-1]
        mean_xyah[:2] = xy
        vxy = np.matrix(np.hstack([mean_xyah[4:6], 0]))
        vxy = vxy * mat
        vxy = np.squeeze(np.array(vxy))[:-1]
        mean_xyah[4:6] = vxy
        strack.mean = mean_xyah
    def orb_map_matrix(self,img,rescale_ratio):
        def get_map_matrix(src_points, dist_points):
            N = src_points.shape[0]
            src_points = np.hstack([np.matrix(src_points.copy()), np.ones((N, 1))])  # shape [N,3]
            dist_points = np.hstack([np.matrix(dist_points.copy()), np.ones((N, 1))])  # shape [N,3]
            translation_matrix = np.linalg.inv(src_points.T * src_points) * src_points.T * dist_points
            return translation_matrix
        def orb_detet(img):
            orb = cv2.ORB_create()
            return orb.detectAndCompute(img, None)
        def sift_detect(img):
            # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            sift = cv2.SIFT_create()
            return sift.detectAndCompute(img,None)
        # kp2,des2 = sift_detect(img)
        kp2,des2 = orb_detet(img)
        if self.last_detect is not None:
            kp1, des1 = self.last_detect

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            goodMatch = sorted(matches, key=lambda x: x.distance)

            # 增加一个维度
            pt1s = []
            pt2s = []
            for match in goodMatch[:20]:
                pt1s.append(kp1[match.queryIdx].pt)
                pt2s.append(kp2[match.trainIdx].pt)
            pt1s = np.array(pt1s)
            pt2s = np.array(pt2s)

            map_matrix = get_map_matrix(pt1s, pt2s)
            left_mat = rescale_ratio * np.matrix(np.eye(3))
            right_mat = (1/rescale_ratio) * np.matrix(np.eye(3))
            left_mat[2, 2] = 1
            right_mat[2, 2] = 1
            map_matrix = left_mat * map_matrix * right_mat
        else:
            map_matrix = np.eye(3)
        self.last_detect = (kp2, des2)
        return map_matrix
    def ecc_map_matrix(self,img,rescale_ratio):
        # Convert images to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.last_img is None:
            self.last_img = img
            return np.matrix(np.eye(3))
        # Define the motion model
        warp_mode = cv2.MOTION_TRANSLATION

        # Define 2x3 or 3x3 matrices and initialize the matrix to identity
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        # Specify the number of iterations.
        number_of_iterations = 100;

        # Specify the threshold of the increment
        # in the correlation coefficient between two iterations
        termination_eps = 1e-2;

        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

        # Run the ECC algorithm. The results are stored in warp_matrix.
        _,map_matrix = cv2.findTransformECC(self.last_img, img, warp_matrix, warp_mode, criteria)

        map_matrix = np.matrix(np.vstack([map_matrix, np.array([0, 0, 1])]).T)
        # print(warp_matrix)
        self.last_img = img
        left_mat = rescale_ratio * np.matrix(np.eye(3))
        right_mat = (1 / rescale_ratio) * np.matrix(np.eye(3))
        left_mat[2, 2] = 1
        right_mat[2, 2] = 1
        return left_mat * map_matrix * right_mat
    def get_two_img_map_matrix(self,img, rescale_ratio):
        if self.use_mat:
            if self.recorder.mode == 'record':
                map_matrix = self.ecc_map_matrix(img,rescale_ratio)
                # map_matrix = self.orb_map_matrix(img,rescale_ratio)
                self.recorder.record_mat(map_matrix.tolist())
            else:
                map_matrix = np.matrix(self.recorder.get_mat())
        else:
            map_matrix = np.eye(3)
        map_matrix = np.eye(3)
        self.map_matrix = self.map_matrix * map_matrix
        self.matrixs.append(self.map_matrix)
        if len(self.window_shake_ratios) == 0:
            self.window_shake_ratios.append(0)
        else:
            self.window_shake_ratios.append(self.window_shake_ratios[-1] + np.abs(map_matrix[2,0]) + np.abs(map_matrix[2,1]))
        return map_matrix
    def insert_frame_lost_track(self,refind_stracks):
        insert_frame = {}
        for strack in refind_stracks:
            mean_shake_ratio = (self.window_shake_ratios[-1] - self.window_shake_ratios[strack.last_track_frame_id-1]) / (strack.frame_id - strack.last_track_frame_id)
            max_insert_frames = self.max_num_insframe * np.exp(-np.abs(strack.mean[4])-np.abs(strack.mean[5])-mean_shake_ratio)
            if strack.frame_id - strack.last_track_frame_id >= max_insert_frames:
                continue

            insert_frame[strack.track_id] = []

            def func(l):
                return 1
            map_mat = np.linalg.inv(self.matrixs[strack.last_track_frame_id-1])*self.matrixs[-1]
            tmp_tlbr = self.compute_mapped_tlbr(strack.tlbr,np.linalg.inv(map_mat))
            for frame in range(strack.last_track_frame_id+1,strack.frame_id):
                tlbr = strack.last_track_tlbr + (frame - strack.last_track_frame_id) / (strack.frame_id  - strack.last_track_frame_id) * (tmp_tlbr - strack.tlbr)
                map_mat = np.linalg.inv(self.matrixs[strack.last_track_frame_id - 1]) * self.matrixs[frame-1]
                tlwh = STrack.tlbr_to_tlwh(self.compute_mapped_tlbr(tlbr,map_mat))
                insert_frame[strack.track_id].append((tlwh,frame))

            map_mat = np.linalg.inv(self.matrixs[strack.last_track_frame_id - 1]) * self.matrixs[-1]
            tmp_tlbr = self.compute_mapped_tlbr(strack.last_track_tlbr, map_mat)
            for i,frame in enumerate(range(strack.frame_id-1,strack.last_track_frame_id,-1)):
                tlbr = strack.tlbr - (strack.frame_id - frame) / (strack.frame_id  - strack.last_track_frame_id) * (strack.tlbr - tmp_tlbr)
                map_mat = np.linalg.inv(self.matrixs[frame-1]) * self.matrixs[self.frame_id-1]
                tlwh = STrack.tlbr_to_tlwh(self.compute_mapped_tlbr(tlbr,np.linalg.inv(map_mat)))
                w1 = func(frame - strack.last_track_frame_id)
                w2 = func(strack.frame_id - frame)
                tlwh = (insert_frame[strack.track_id][-1-i][0]*w1 + tlwh*w2) / (w1 + w2)
                insert_frame[strack.track_id][-1 - i] = (tlwh,frame)


            strack.last_track_frame_id = strack.frame_id
            strack.last_track_tlbr = strack.tlbr
        return  insert_frame
    def match(self,tracks,dets,feature,threshold,activated_starcks,refind_stracks):
        if len(tracks) == 0 or len(dets) == 0:
            return tracks,dets
        if feature == 'Iou':
            dists = matching.iou_distance(tracks, dets)
        elif feature == 'embedding':
            dists = matching.embedding_distance(tracks, dets)
            dists = matching.fuse_motion(self.kalman_filter, dists, tracks, dets)
        else:
            exit()
            return
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=threshold)
        iou_dist = matching.iou_distance(tracks, dets)
        for itracked, idet in matches:
            track = tracks[itracked]
            det = dets[idet]
            self.iou_mean = self.iou_mean * self.match_num
            self.iou_mean += iou_dist[itracked,idet]
            self.match_num += 1
            self.iou_mean /= self.match_num
            if track.state == TrackState.Tracked:
                track.update(dets[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                x = (det.tlbr[0] + det.tlbr[2]) / 2
                w = det.tlwh[2]
                y = (det.tlbr[1] + det.tlbr[3]) / 2
                x1 = (track.tlbr[0] + track.tlbr[2]) / 2
                y1 = (track.tlbr[1] + track.tlbr[3]) / 2
                import math
                dis = math.sqrt((x1-x)**2 + (y1 - y)**2)
                self.iou_dist_time[(self.frame_id - track.last_track_frame_id)//5+1].append(dis)
                if self.frame_id - track.last_track_frame_id <= 5:
                    self.lost_det += 1
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        return [tracks[i] for i in u_track],[dets[i] for i in u_detection if i not in []]

    def __del__(self):
        tmp = 0
        for i in range(len(self.iou_dist_time)):
            if len(self.iou_dist_time[i+1]) != 0:
                print(sum(self.iou_dist_time[i+1])/len(self.iou_dist_time[i+1]))
                tmp = sum(self.iou_dist_time[i+1])/len(self.iou_dist_time[i+1])
            else:
                print(tmp)
        print('='*100)
        for i in range(len(self.iou_dist_time)):
            print((i+1)*5)
    def record_feat(self):
        import json
        f = open('feat.json','w')
        json.dump(self.feat_record,f)
        f.close()
def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb


def remove_fp_stracks(stracksa, n_frame=10):
    remain = []
    for t in stracksa:
        score_5 = t.score_list[-n_frame:]
        score_5 = np.array(score_5, dtype=np.float32)
        index = score_5 < 0.45
        num = np.sum(index)
        if num < n_frame:
            remain.append(t)
    return remain



