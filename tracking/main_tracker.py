import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from copy import deepcopy
import cv2
from tracking.track_state import trackState
from utils.tools import iou, nms, normalization, separate_measure
from dnn.neural_net import neuralNet


class track:
    def __init__(self, seq_name, seq_info, data, config, semi_on, fr_delay, visualization=False):
        # Save recent 5-images to save the processing time of new hyp init
        self.imgs = []

        self.trk_result = []
        self.trk_state = []
        self.hyp_dets = []
        self.hyp_valid = []
        self.hyp_assoc = []
        self.max_id = 1

        self.data = data
        self.seq_name = seq_name
        self.img_shp = seq_info[0:2]
        self.fps = seq_info[2]
        self.config = config
        self.NN = neuralNet(is_test=True)

        self.semi_on = semi_on
        self.fr_delay = fr_delay
        self.visualization = visualization

    def __del__(self):
        print("track deleted")

    def track(self, bgr_img, dets, fr_num):
        # dets : [[x,y,w,h,conf], ..., [x,y,w,h,conf]]
        dets = self.det_preprocessing(dets, fr_num)

        if len(self.imgs) > self.config.max_hyp_len:
            self.imgs.pop(0)
        self.imgs.append(bgr_img)

        # Track-detection association
        prev_trk = deepcopy(self.trk_state)

        if len(prev_trk) > 0 and len(dets) > 0:

            # Get detection templates
            det_templates = np.zeros((dets.shape[0], 128, 64, 3))
            for idx, det in enumerate(dets):
                cropped = bgr_img[det[1]:det[1] + det[3], det[0]:det[0] + det[2]]
                det_templates[idx, :, :, :] = cv2.resize(cropped, (64, 128))

            # Construct a similarity matrix
            gating_mat = np.zeros((len(dets), len(prev_trk)))
            sim_mat = np.zeros((len(dets), len(prev_trk)))

            # Motion similarity calculation
            for i, det in enumerate(dets):
                for j, trk in enumerate(prev_trk):
                    mot_sim = trk.mahalanobis_distance(det, fr_num)
                    gating_mat[i, j] = mot_sim * prev_trk[j].get_shp_similarity(dets[i][2:4], fr_num)
                    sim_mat[i, j] = mot_sim

            # Appearance similarity calculation
            # Pre-calculate the Neural-Net input shape to save the time
            num_matching_templates = []
            for i in range(gating_mat.shape[0]):
                for j in range(gating_mat.shape[1]):
                    if gating_mat[i, j] > self.config.gating_thresh:
                        num_matching_templates.append(len(prev_trk[j].historical_app) + 1)
                    else:
                        num_matching_templates.append(0)

            if sum(num_matching_templates) > 0:
                input_templates = np.zeros((sum(num_matching_templates), 128, 64, 6))
                input_shps = np.zeros((sum(num_matching_templates), 3))

                # Make input batch
                cur_idx = 0
                accum_num = 0
                for i in range(gating_mat.shape[0]):
                    for j in range(gating_mat.shape[1]):
                        if gating_mat[i, j] > self.config.gating_thresh:
                            matching_tmpls = []
                            matching_shps = []
                            anchor_shp = np.array([fr_num, *list(dets[i][2:4])], dtype=float)

                            # Historical appearances
                            for trk_tmpl, trk_shp, trk_fr in zip(prev_trk[j].historical_app, prev_trk[j].historical_shps, prev_trk[j].historical_frs):
                                self.accumulate_app_pairs(matching_tmpls, trk_tmpl, det_templates[i])
                                self.accumulate_shp_pairs(matching_shps, trk_shp, trk_fr, anchor_shp)

                            # Recent appearance
                            self.accumulate_app_pairs(matching_tmpls, prev_trk[j].recent_app, det_templates[i])
                            self.accumulate_shp_pairs(matching_shps, prev_trk[j].recent_shp, prev_trk[j].recent_fr, anchor_shp)

                            input_templates[accum_num:accum_num+num_matching_templates[cur_idx], :, : , :] = np.array(matching_tmpls)
                            input_shps[accum_num:accum_num+num_matching_templates[cur_idx], :] = np.array(matching_shps)
                            accum_num += num_matching_templates[cur_idx]
                        cur_idx += 1

                # JI-Net based matching-feature extraction
                feature_batch = self.NN.get_feature(input_templates)

                # Create LSTM input batch
                valid_matching_templates = [i for i in num_matching_templates if i > 0]

                input_batch = np.zeros((sum(num_matching_templates), self.config.max_hist_len+1, 150+3))
                cur_idx = 0

                for idx, tmp_num in enumerate(valid_matching_templates):
                    if tmp_num > 0:
                        tmp_features = feature_batch[cur_idx:cur_idx + tmp_num]
                        input_batch[idx, self.config.max_hist_len+1 - tmp_num:, :-3] = tmp_features
                        input_batch[idx, self.config.max_hist_len+1 - tmp_num:, -3:] = input_shps[cur_idx:cur_idx + tmp_num]
                        cur_idx += tmp_num

                # Final appearance likelihood from Deep-TAMA
                likelihoods = self.NN.get_likelihood(input_batch)

                cur_idx = 0
                for i in range(gating_mat.shape[0]):
                    for j in range(gating_mat.shape[1]):
                        if gating_mat[i, j] > self.config.gating_thresh:
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
                        # Strict initialization
                        if iou(det, hyp) > self.config.assoc_iou_thresh:
                            is_assoc = True
                            tmp_assoc.append(hyp_ind)
                    if is_assoc:
                        dets_unassoc[det_ind] = False
                        tmp_hyp_dets.append(det)
                        tmp_hyp_valid.append(True)
                        tmp_hyp_assoc.append(tmp_assoc)
                    else:
                        for hyp_ind, hyp in enumerate(prev_hyp):
                            # Weak initialization
                            pos_dist, shp_dist = separate_measure(det, hyp)
                            if pos_dist < det[3]*self.config.assoc_dist_thresh and shp_dist > self.config.assoc_shp_thresh:
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
                for tail_idx, tail_assoc in enumerate(self.hyp_assoc[-1]):
                    for tail_assoc_idx in tail_assoc:
                        tmp_trk = [tail_assoc_idx]
                        out_trk = self.recursive_find(self.config.max_hyp_len-1, tail_assoc_idx, tmp_trk)
                        if out_trk:
                            out_trk.append(tail_idx)
                            tmp_to_track = []
                            for depth, hyp_idx in enumerate(out_trk):
                                self.hyp_valid[depth][hyp_idx] = False
                                tmp_to_track.append(self.hyp_dets[depth][hyp_idx])
                            to_tracks.append(tmp_to_track)

            # Recursively update trk
            if len(to_tracks) > 0:
                for to_track in to_tracks:
                    for i, y in enumerate(to_track):
                        tmp_fr = fr_num - self.config.max_hyp_len + i
                        template = cv2.resize(self.imgs[i][y[1]:y[1] + y[3], y[0]:y[0] + y[2]], (64, 128))
                        if i == 0:
                            tmp_trk = trackState(y, template, tmp_fr, self.max_id, self.config)
                            self.max_id += 1
                        else:
                            tmp_trk.predict(tmp_fr, self.config)
                            tmp_trk.update(y, template, self.config.hist_thresh, tmp_fr, self.config, is_init=True)

                        # Initialization hypothesis restoration (semi online mode only)
                        if self.semi_on and i < self.config.max_hyp_len:
                            tmp_state = [tmp_trk.track_id, tmp_trk.X[0][0], tmp_trk.X[2][0], tmp_trk.recent_shp[0],
                                         tmp_trk.recent_shp[1], tmp_trk.color]
                            self.trk_result[fr_num - (self.config.max_hyp_len-i) - 1].append(tmp_state)

                    self.trk_state.append(tmp_trk)
        else:
            self.hyp_dets.append([])
            self.hyp_valid.append([])
            self.hyp_assoc.append([])

        # Init hyp
        if len(dets) > 0:
            unassoc_dets = dets[dets_unassoc]
            for unassoc_det in unassoc_dets:
                self.hyp_dets[-1].append(unassoc_det)
                self.hyp_valid[-1].append(True)
                self.hyp_assoc[-1].append([])

        # Track termination
        prev_trk_len = len(self.trk_state)
        for i, trk in enumerate(self.trk_state[::-1]):
            if (fr_num - trk.recent_fr) > self.config.miss_thresh:
                self.trk_state.pop(prev_trk_len - i - 1)
            elif trk.X[0, 0] <= 0 or trk.X[2, 0] <= 0 or trk.X[0, 0] >= self.img_shp[0] or trk.X[2, 0] >= self.img_shp[1]:
                self.trk_state.pop(prev_trk_len - i - 1)

        # Track save
        self.track_save(fr_num)

        # Track interpolation (semi online mode only)
        self.track_interpolation(fr_num)

        # Next frame prediction
        for trk in self.trk_state:
            trk.predict(fr_num+1, self.config)

    def accumulate_shp_pairs(self, to_list, trk_shp, trk_fr, anchor_shp):
        trk_shp = np.array([trk_fr, *list(trk_shp)])
        matching_shp = trk_shp - anchor_shp
        matching_shp[0] /= self.fps
        matching_shp[1:3] /= anchor_shp[1:3]
        to_list.append(matching_shp)

    def accumulate_app_pairs(self, to_list, app_pair1, app_pair2):
        matching_tmpl = np.concatenate((normalization(app_pair1), normalization(app_pair2)), 2)
        to_list.append(matching_tmpl)

    def det_preprocessing(self, dets, fr_num):
        if len(dets) > 0:
            dets = dets[:, 1:6]
            for i in range(dets.shape[0]):
                dets[i][dets[i] < 0] = 0
            dets = np.hstack((dets, np.ones((dets.shape[0], 1)) * fr_num))
            keep = nms(dets, self.config.nms_iou_thresh)
            dets = dets[keep]
            dets = dets[dets[:, 5] > self.config.det_thresh].astype(int)
        else:
            dets = []

        return dets

    def recursive_find(self, _depth, assoc_idx, _tmp_trk):
        if _depth == 0:
            if self.hyp_valid[_depth][assoc_idx]:
                return _tmp_trk
            else:
                return []

        for next_idx in self.hyp_assoc[_depth][assoc_idx]:
            if self.hyp_valid[_depth - 1][next_idx]:
                _tmp_trk.insert(0, next_idx)
                if self.recursive_find(_depth - 1, next_idx, _tmp_trk):
                    return _tmp_trk
                else:
                    _tmp_trk.pop(0)
        return []

    def track_interpolation(self, cur_fr):
        # Interpolate holes within an each track
        for anchor_trk in self.trk_result[-1]:
            if cur_fr <= self.fr_delay:
                continue

            # Find same id in fr_delay range
            found = False
            found_i = 0
            found_trk = None
            i = 1
            while not found and i <= self.fr_delay:
                for refer_trk in self.trk_result[-1-i]:
                    if refer_trk[0] == anchor_trk[0]:
                        found = True
                        found_i = i
                        found_trk = refer_trk
                        break
                i += 1

            # Make interpolation
            if found and found_i > 1:
                x_gap = (anchor_trk[1] - found_trk[1]) / found_i
                y_gap = (anchor_trk[2] - found_trk[2]) / found_i
                w_gap = (anchor_trk[3] - found_trk[3]) / found_i
                h_gap = (anchor_trk[4] - found_trk[4]) / found_i
                intp_trk = []
                for i in range(1, found_i):
                    tmp_trk = deepcopy(found_trk)
                    tmp_trk[1] += i*x_gap
                    tmp_trk[2] += i*y_gap
                    tmp_trk[3] += i*w_gap
                    tmp_trk[4] += i*h_gap
                    tmp_trk[5] = anchor_trk[5]
                    intp_trk.append(tmp_trk)

                for i in range(1, found_i):
                    self.trk_result[-1-i].append(intp_trk[-i])

    def track_save(self, fr_num):
        tmp_save = []
        valid_miss_num = 1
        if len(self.trk_state) > 0:
            for trk in self.trk_state:
                if trk.recent_fr > fr_num - valid_miss_num:
                    tmp_state = [trk.track_id, trk.X[0][0], trk.X[2][0], trk.get_shp(fr_num)[0], trk.get_shp(fr_num)[1], trk.color]
                    tmp_save.append(tmp_state)

        self.trk_result.append(tmp_save)