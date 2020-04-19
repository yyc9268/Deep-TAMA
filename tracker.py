import os
import numpy as np
from scipy.optimize import linear_sum_assignment
import dataLoader as ds
from copy import deepcopy
import cv2
from kalmanFilter import TrackState
from Tools import IOU, NMS
from Config import Config


class Track:
    def __init__(self, _seq_info, _config, visualization=False):
        self.trk_result = []
        self.trk_state = []
        self.hyp_dets = []
        self.hyp_valid = []
        self.hyp_assoc = []
        self.max_id = 1

        self.img_shp = _seq_info[0:2]
        self.fps = _seq_info[2]
        self.config = _config
        self.visualization = visualization

    def track(self, bgr_img, dets, fr_num):
        # dets : [[x,y,w,h,conf], ..., [x,y,w,h,conf]]
        dets = self.det_preprocessing(dets, fr_num)

        # Track-detection association
        prev_trk = deepcopy(self.trk_state)

        if len(prev_trk) > 0 and len(dets) > 0:

            # Construct a similarity matrix
            sim_mat = np.zeros((len(dets), len(prev_trk)))

            for i, det in enumerate(dets):
                for j, trk in enumerate(prev_trk):
                    mot_sim = trk.mahalanobis_distance(det)
                    sim_mat[i, j] = mot_sim

            # hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(-sim_mat)

            # Track update
            assoc_det_ind = []
            for trk_ind, det_ind in zip(col_ind, row_ind):
                if sim_mat[det_ind, trk_ind] > self.config.assoc_thresh:
                    y = dets[det_ind]
                    template = bgr_img[y[1], y[1] + y[3], y[0], y[0] + y[2]]
                    self.trk_state[trk_ind].update(dets[det_ind], template, sim_mat[det_ind, trk_ind], fr_num, self.config)
                    assoc_det_ind.append(det_ind)

            # Associated det removal
            dets = np.delete(dets, assoc_det_ind, axis=0)

        # Track initialization
        dets_unassoc = [True] * len(dets)
        if len(self.hyp_dets) > self.config.max_hyp_len:
            self.hyp_dets.pop(0)
            self.hyp_valid.pop(0)
            self.hyp_assoc.pop(0)

        if len(self.hyp_dets) > 0 and len(dets) > 0:
            prev_hyp = deepcopy(self.hyp_dets[-1])

            tmp_hyp_dets = []
            tmp_hyp_valid = []
            tmp_hyp_assoc = []
            if len(dets) > 0 and len(prev_hyp) > 0:
                for det_ind, det in enumerate(dets):
                    is_assoc = False
                    tmp_assoc = []
                    for hyp_ind, hyp in enumerate(prev_hyp):
                        # Hierarchical initialization
                        if IOU(det, hyp) > self.config.assoc_iou_thresh:
                            is_assoc = True
                            tmp_assoc.append(hyp_ind)
                    if is_assoc:
                        dets_unassoc[det_ind] = False
                        tmp_hyp_dets.append(det)
                        tmp_hyp_valid.append(True)
                        tmp_hyp_assoc.append(tmp_assoc)

            self.hyp_dets.append(tmp_hyp_dets)
            self.hyp_valid.append(tmp_hyp_valid)
            self.hyp_assoc.append(tmp_hyp_assoc)

            # Init trk
            to_tracks = []
            if len(self.hyp_dets) > self.config.max_hyp_len:
                for tail_assoc in self.hyp_assoc[-1]:
                    for tail_assoc_idx in tail_assoc:
                        tmp_trk = [tail_assoc_idx]
                        out_trk = self.recursive_find(self.config.max_hyp_len-1, tail_assoc_idx, tmp_trk)
                        if out_trk:
                            tmp_to_track = []
                            for depth, hyp_idx in enumerate(out_trk):
                                self.hyp_valid[depth][hyp_idx] = False
                                tmp_to_track.insert(0, self.hyp_dets[depth][hyp_idx])
                            to_tracks.append(tmp_to_track)

            # Recursively update trk
            for to_track in to_tracks:
                for i, y in enumerate(to_track):
                    tmp_fr = fr_num - self.config.max_hyp_len + i
                    template = bgr_img[y[1]:y[1] + y[3], y[0]:y[0] + y[2]]
                    if i == 0:
                        tmp_state = TrackState(y, tmp_fr, template, self.max_id)
                        self.max_id += 1
                    else:
                        tmp_state.predict(tmp_fr, self.config)
                        tmp_state.update(y, template, None, tmp_fr, self.config)

                self.trk_state.append(tmp_state)

        else:
            self.hyp_dets.append([])
            self.hyp_valid.append([])
            self.hyp_assoc.append([])

        # Init hyp
        unassoc_dets = dets[dets_unassoc]
        for unassoc_det in unassoc_dets:
            self.hyp_dets[-1].append(unassoc_det)
            self.hyp_valid[-1].append(True)
            self.hyp_assoc[-1].append([])

        # Track termination
        prev_trk_len = len(self.trk_state)
        for i, trk in enumerate(self.trk_state[::-1]):
            if fr_num - trk.recent_fr > self.config.miss_thresh:
                self.trk_state.pop(prev_trk_len - i - 1)
            elif trk.X[0, 0] <= 0 or trk.X[2, 0] <= 0 or trk.X[0, 0] >= seq_info[0] or trk.X[2, 0] >= seq_info[1]:
                self.trk_state.pop(prev_trk_len - i - 1)

        # Track save
        self.track_save(fr_num)

        # Track visualization
        if self.visualization:
            self.track_visualization(bgr_img)

        # Next frame prediction
        for trk in self.trk_state:
            trk.predict(fr_num+1, self.config)

    def det_preprocessing(self, dets, fr_num):
        dets = dets[:, 1:6]
        dets = np.hstack((dets, np.ones((dets.shape[0], 1)) * fr_num))
        keep = NMS(dets, self.config.nms_iou_thresh)
        dets = dets[keep]
        dets = dets[dets[:, 5] > self.config.det_thresh].astype(int)

        return dets

    def recursive_find(self, _depth, assoc_idx, _tmp_trk):
        if _depth == 0:
            if self.hyp_valid[_depth][assoc_idx]:
                return _tmp_trk
            else:
                return []

        for next_idx in self.hyp_assoc[_depth][assoc_idx]:
            if self.hyp_valid[_depth - 1][next_idx] and self.hyp_assoc[_depth - 1][next_idx]:
                _tmp_trk.insert(0, next_idx)
                if self.recursive_find(_depth - 1, next_idx, _tmp_trk):
                    return _tmp_trk
                else:
                    _tmp_trk.pop(0)
        return []

    def track_save(self, fr_num):
        tmp_save = []
        valid_miss_num = 5
        if len(self.trk_state) > 0:
            for trk in self.trk_state:
                if trk.recent_fr > fr_num - valid_miss_num:
                    tmp_X = list(map(float, trk.X))
                    tmp_save.append(tmp_X)

        self.trk_result.append(tmp_save)

    def track_write(self):
        return NotImplementedError

    def track_visualization(self, bgr_img):
        if len(self.trk_result[-1]) > 0:
            for trk in self.trk_result[-1]:
                tmp_X = list(map(int, trk))
                cv2.putText(bgr_img, str(trk.track_id), (tmp_X[0], tmp_X[2]), cv2.FONT_HERSHEY_COMPLEX, 2,
                            trk.color, 2)
                cv2.rectangle(bgr_img, (tmp_X[0], tmp_X[2]), (tmp_X[0] + trk.shape[0], tmp_X[2] + trk.shape[1]),
                              trk.color, 3)


if __name__=="__main__":

    seq_names = ["MOT16-02"]
    data = ds.data(is_test=True)

    for seq_name in seq_names:
        if not os.path.exists(seq_name):
            os.mkdir(seq_name)

        # get sequence info
        seq_info = data.get_seq_info(seq_name=seq_name)

        # get tracking parameters
        config = Config(seq_info[-1])

        track = Track(seq_info, config, visualization=True)
        fr_end = 10000
        for cur_fr in range(1, fr_end):
            _bgr_img, _dets = data.get_frame_info(seq_name="MOT16-02", frame_num=cur_fr)
            track.track(_bgr_img, _dets, cur_fr)