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
    with torch.no_grad():
        output = self.model(im_blob)[-1]
        hm = output['hm'].sigmoid_()
        wh = output['wh']
        id_feature = output['id']
        id_feature = F.normalize(id_feature, dim=1)

        reg = output['reg'] if self.opt.reg_offset else None
        dets, inds = mot_decode(hm, wh, reg=reg, ltrb=self.opt.ltrb, K=self.opt.K)
        id_feature = _tranpose_and_gather_feat(id_feature, inds)
        id_feature = id_feature.squeeze(0)
        id_feature = id_feature.cpu().numpy()

    dets = self.post_process(dets, meta)
    dets = self.merge_outputs([dets])[1]

    remain_inds = dets[:, 4] > self.opt.conf_thres
    inds_low = dets[:, 4] > 0.2
    inds_high = dets[:, 4] < self.opt.conf_thres
    inds_second = np.logical_and(inds_low, inds_high)


    """compute the map matrix"""
    rescale_ratio = 0.25
    img0 = cv2.resize(img0, None, None, rescale_ratio, rescale_ratio)
    self.get_two_img_map_matrix(img0, rescale_ratio)
    inv_map_matrix = np.linalg.inv(
        np.linalg.inv(self.matrixs[max(-self.window_matrix - 1, -len(self.matrixs))]) * self.map_matrix)
    for i in range(len(dets)):
        tlbr = self.compute_mapped_tlbr(dets[i, :4], inv_map_matrix)
        dets[i,:4] = tlbr
    """compute the map matrix"""



    dets_second = dets[inds_second]
    id_feature_second = id_feature[inds_second]
    if len(dets_second) > 0:
        '''Detections'''
        detections_second = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30) for
                      (tlbrs, f) in zip(dets_second[:, :5], id_feature_second)]
    else:
        detections_second = []

    dets = dets[remain_inds]
    id_feature = id_feature[remain_inds]
    if len(dets) > 0:
        '''Detections'''
        detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30) for
                      (tlbrs, f) in zip(dets[:, :5], id_feature)]
    else:
        detections = []



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
    """compute the map matrix"""
    for strack in strack_pool:
        if len(self.matrixs) > self.window_matrix + 1:
            self.compute_mapped_track(strack, np.linalg.inv(self.matrixs[-self.window_matrix - 2]) * self.matrixs[
                -self.window_matrix - 1])
    """compute the map matrix"""



    ''' Step 2: First association, with embedding'''
    dists = matching.embedding_distance(tracked_stracks, detections)
    ioudists = matching.iou_distance(tracked_stracks,detections)
    dists[ioudists>0.8] = np.inf
    dists = matching.fuse_motion(self.kalman_filter, dists, tracked_stracks, detections)
    matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.4)

    for itracked, idet in matches:
        track = tracked_stracks[itracked]
        det = detections[idet]
        if track.state == TrackState.Tracked:
            track.update(detections[idet], self.frame_id)
            activated_starcks.append(track)
        else:
            track.re_activate(det, self.frame_id, new_id=False)
            refind_stracks.append(track)
    ''' Step 2: First association, with embedding'''

    ''' Step 4: Second association, with IOU'''
    detections = [detections[i] for i in u_detection]
    dists = matching.embedding_distance(self.lost_stracks, detections)
    ioudists = matching.iou_distance(self.lost_stracks, detections)
    dists[ioudists > 0.8] = np.inf
    matches, _, u_detection = matching.linear_assignment(dists, thresh=0.4)

    for itracked, idet in matches:
        track = self.lost_stracks[itracked]
        det = detections[idet]
        if track.state == TrackState.Tracked:
            track.update(detections[idet], self.frame_id)
            activated_starcks.append(track)
        else:
            track.re_activate(det, self.frame_id, new_id=False)
            refind_stracks.append(track)
    ''' Step 4: Second association, with IOU'''

    ''' Step 3: Second association, with IOU'''
    detections = [detections[i] for i in u_detection]
    r_tracked_stracks = [tracked_stracks[i] for i in u_track if tracked_stracks[i].state == TrackState.Tracked]
    dists = matching.iou_distance(r_tracked_stracks, detections)
    matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)

    for itracked, idet in matches:
        track = r_tracked_stracks[itracked]
        det = detections[idet]
        if track.state == TrackState.Tracked:
            track.update(det, self.frame_id)
            activated_starcks.append(track)
        else:
            track.re_activate(det, self.frame_id, new_id=False)
            refind_stracks.append(track)
    ''' Step 3: Second association, with IOU'''





    ''' Step 5: association whit IOU on low score detection'''
    second_tracked_stracks = [r_tracked_stracks[i] for i in u_track if r_tracked_stracks[i].state == TrackState.Tracked]
    dists = matching.iou_distance(second_tracked_stracks, detections_second)
    matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.4)
    for itracked, idet in matches:
        track = second_tracked_stracks[itracked]
        det = detections_second[idet]
        if track.state == TrackState.Tracked:
            track.update(det, self.frame_id)
            activated_starcks.append(track)
        else:
            track.re_activate(det, self.frame_id, new_id=False)
            refind_stracks.append(track)
    ''' Step 5: association whit IOU on low score detection'''


    for it in u_track:
        #track = r_tracked_stracks[it]
        track = second_tracked_stracks[it]
        if not track.state == TrackState.Lost:
            track.mark_lost()
            lost_stracks.append(track)

    '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
    detections = [detections[i] for i in u_detection]
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


    return out, self.insert_frame_lost_track(refind_stracks)