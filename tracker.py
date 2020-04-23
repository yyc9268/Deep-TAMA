import os
import numpy as np
from scipy.optimize import linear_sum_assignment
import dataLoader as ds
from copy import deepcopy
import cv2
from kalmanFilter import TrackState
from Tools import IOU, NMS
from Config import Config
from TrainDeepDA import NeuralNet

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
        self.NN = NeuralNet(is_test=True)

    def track(self, bgr_img, dets, fr_num):
        # dets : [[x,y,w,h,conf], ..., [x,y,w,h,conf]]
        dets = self.det_preprocessing(dets, fr_num)
        print(dets.shape)

        # Track-detection association
        prev_trk = deepcopy(self.trk_state)

        if len(prev_trk) > 0 and len(dets) > 0:

            # Get detection templates
            det_templates = np.zeros((dets.shape[0], 128, 64, 3))
            for idx, det in enumerate(dets):
                cropped = bgr_img[det[1]:det[1] + det[3], det[0]:det[0] + det[2]]
                det_templates[idx, :, :, :] = cv2.resize(cropped, (64, 128))

            # Construct a similarity matrix
            sim_mat = np.zeros((len(dets), len(prev_trk)))

            # Motion similarity calculation
            for i, det in enumerate(dets):
                for j, trk in enumerate(prev_trk):
                    mot_sim = trk.mahalanobis_distance(det)
                    sim_mat[i, j] = mot_sim

            # Appearance similarity calculation
            # Pre-calculate the Neural-Net input shape to save the time
            num_matching_templates = []
            for i in range(sim_mat.shape[0]):
                for j in range(sim_mat.shape[1]):
                    if sim_mat[i, j] > config.assoc_thresh:
                        num_matching_templates.append(len(prev_trk[j].historical_app) + 1)
                    else:
                        num_matching_templates.append(0)

            input_templates = np.zeros((sum(num_matching_templates), 128, 64, 6))
            input_shps = np.zeros((sum(num_matching_templates), 3))

            # Make input batch
            cur_idx = 0
            accum_num = 0
            for i in range(sim_mat.shape[0]):
                for j in range(sim_mat.shape[1]):
                    if sim_mat[i, j] > config.assoc_thresh:
                        matching_tmpls = []
                        matching_shps = []
                        anchor_shp = np.array([fr_num, *list(dets[i][2:4])], dtype=float)

                        # Historical appearances
                        for trk_tmpl, trk_shp, trk_fr in zip(prev_trk[j].historical_app, prev_trk[j].historical_shps, prev_trk[j].historical_frs):
                            matching_tmpl = np.concatenate((trk_tmpl, det_templates[i]), 2)
                            matching_tmpls.append(matching_tmpl)
                            trk_shp = np.array([trk_fr, *list(trk_shp)])
                            matching_shp = trk_shp - anchor_shp
                            matching_shp[0] /= self.fps
                            matching_shp[1:3] /= anchor_shp[1:3]
                            matching_shps.append(matching_shp)

                        # Recent appearance
                        matching_tmpls.append(np.concatenate((prev_trk[j].recent_app, det_templates[i]), 2))
                        trk_shp = np.array([prev_trk[j].recent_fr, *list(prev_trk[j].recent_shp)], dtype=float)
                        matching_shp = trk_shp - anchor_shp
                        matching_shp[0] /= self.fps
                        matching_shp[1:3] /= anchor_shp[1:3]
                        matching_shps.append(matching_shp)

                        input_templates[accum_num:num_matching_templates[cur_idx], :, : , :] = np.array(matching_tmpls)
                        input_shps[accum_num:num_matching_templates[cur_idx], :] = np.array(matching_shps)
                        accum_num += 1
                    cur_idx += 1

            # JI-Net based matching-feature extraction
            feature_batch = self.NN.getFeature(input_templates)

            # Create LSTM input batch
            valid_matching_templates = [i for i in num_matching_templates if i>0]

            input_batch = np.zeros((sum(num_matching_templates), self.config.max_hist_len+1, 150+3,))
            cur_idx = 0

            for idx, tmp_num in enumerate(valid_matching_templates):
                if tmp_num > 0:
                    tmp_features = feature_batch[cur_idx:cur_idx + tmp_num]
                    input_batch[idx, self.config.max_hist_len+1 - tmp_num:, :-3] = tmp_features
                    input_batch[idx, self.config.max_hist_len+1 - tmp_num:, -3:] = input_shps[cur_idx:cur_idx + tmp_num]
                    cur_idx += tmp_num

            # Final appearance likelihood from Deep-TAMA
            likelihoods = self.NN.getLikelihood(input_batch)

            cur_idx = 0
            for i in range(sim_mat.shape[0]):
                for j in range(sim_mat.shape[1]):
                    if sim_mat[i, j] > config.assoc_thresh:
                        sim_mat[i, j] *= likelihoods[cur_idx][0]
                        cur_idx += 1

            # hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(-sim_mat)

            # Track update
            assoc_det_ind = []
            for trk_ind, det_ind in zip(col_ind, row_ind):
                if sim_mat[det_ind, trk_ind] > self.config.assoc_thresh:
                    y = dets[det_ind]
                    template = cv2.resize(bgr_img[y[1]:y[1] + y[3], y[0]: y[0] + y[2]], (64, 128))
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
                    template = cv2.resize(bgr_img[y[1]:y[1] + y[3], y[0]:y[0] + y[2]], (64, 128))
                    if i == 0:
                        tmp_state = TrackState(y, template, tmp_fr, self.max_id, self.config)
                        self.max_id += 1
                    else:
                        tmp_state.predict(tmp_fr, self.config)
                        tmp_state.update(y, template, self.config.hist_thresh, tmp_fr, self.config)

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
                    tmp_save.append(trk)

        self.trk_result.append(tmp_save)

    def track_write(self):
        return NotImplementedError

    def track_visualization(self, bgr_img):
        if len(self.trk_result[-1]) > 0:
            for trk in self.trk_result[-1]:
                cv2.putText(bgr_img, str(trk.track_id), (int(trk.X[0]), int(trk.X[2])), cv2.FONT_HERSHEY_COMPLEX, 2,
                            trk.color, 2)
                cv2.rectangle(bgr_img, (int(trk.X[0]), int(trk.X[2])), (int(trk.X[0] + trk.recent_shp[0]), int(trk.X[2] + trk.recent_shp[1])),
                              trk.color, 3)
        cv2.imshow('{}'.format(cur_fr), bgr_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


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