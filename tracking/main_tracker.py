from copy import deepcopy

import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2

from tracking.track_state import TrackState
from utils.tools import iou, det_preprocessing, normalization, separate_measure
from dnn.neural_net import neuralNet


class Track:
    """
    Main tracking class to perform tracking
    """
    def __init__(self, seq_name, seq_info, data, config, fr_delay, visualization=False):
        self.imgs = []

        self.max_id = 1
        self.trk_result = []
        self.trk_state = []

        # Used for MHT
        self.hyp_dets = []
        self.hyp_valid = []
        self.hyp_assoc = []

        # Used for delayed activate
        self.hyp_state = []

        self.data = data
        self.seq_name = seq_name
        self.img_shp = seq_info[0:2]
        self.fps = seq_info[2]
        self.config = config
        self.NN = neuralNet(is_test=True, train_mode='', config=config)

        self.semi_on = config.semi_on
        self.fr_delay = fr_delay
        self.visualization = visualization

    def __del__(self):
        print("track deleted")

    def track(self, bgr_img, dets, fr_num):
        """
        Do tracking
        :param bgr_img: bgr_img of current frame
        :param dets: [[x,y,w,h,conf], ..., [x,y,w,h,conf]]
        :param fr_num: current frame
        :return: None
        """
        dets = det_preprocessing(dets, fr_num, self.config.nms_iou_thresh, self.config.det_thresh)
        self.trk_result.append([])

        if len(self.imgs) > self.config.max_hyp_len:
            self.imgs.pop(0)
        self.imgs.append(bgr_img)

        # Track-detection association
        if len(self.trk_state) > 0 and len(dets) > 0:
            dets = self.association(dets, self.trk_state, bgr_img, fr_num)

        # Track initialization
        self.initialization(dets, bgr_img, fr_num)

        # Track termination
        prev_trk_len = len(self.trk_state)
        for i, trk in enumerate(self.trk_state[::-1]):
            if (fr_num - trk.recent_fr) >= self.config.miss_thresh:
                self.trk_state.pop(prev_trk_len - i - 1)
            elif trk.X[0, 0] <= 0 or trk.X[2, 0] <= 0 or trk.X[0, 0] >= self.img_shp[0] or trk.X[2, 0] >= self.img_shp[1]:
                self.trk_state.pop(prev_trk_len - i - 1)

        # Track save
        self.track_save(fr_num)

        # Track interpolation (semi online mode only)
        if self.semi_on:
            self.track_interpolation(fr_num)

        # Next frame prediction
        for trk in self.trk_state:
            trk.predict(fr_num+1, self.config)

    def association(self, dets, trk, bgr_img, fr_num):
        """
        Matching between dets and tracks(hyps)
        """
        # Construct a similarity matrix
        sim_mat, num_matching_templates = self.get_geo_similarity(dets, trk, fr_num)

        # Appearance similarity calculation
        if sum(num_matching_templates) > 0:
            likelihoods = self.get_appearance_similarity(dets, trk, sim_mat, num_matching_templates, bgr_img,
                                                         fr_num)

            cur_idx = 0
            for i in range(sim_mat.shape[0]):
                for j in range(sim_mat.shape[1]):
                    if sim_mat[i, j] > 0:
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
                trk[trk_ind].update(dets[det_ind], template, sim_mat[det_ind, trk_ind], fr_num, self.config)
                assoc_det_ind.append(det_ind)

        # Associated det removal
        dets = np.delete(dets, assoc_det_ind, axis=0)

        return dets

    def get_geo_similarity(self, dets, prev_trk, fr_num):
        """
        Calculate geometric likelihoods and return gated matrix
        :param dets: detections
        :param prev_trk: existing tracks
        :param fr_num: current frame number
        :return: likelihood matrix, list containing sequence length of each track
        """
        sim_mat = np.zeros((len(dets), len(prev_trk)))
        num_matching_templates = []

        # Motion similarity calculation
        for i, det in enumerate(dets):
            for j, trk in enumerate(prev_trk):
                if self.config.gating_method == 'iou':
                    mot_sim = iou(det, trk.bbox_info(fr_num))
                    gated = mot_sim < self.config.gating_thresh
                elif self.config.gating_method == 'mahalanobis':
                    mot_sim = trk.mahalanobis_similarity(det)
                    shp_sim = trk.get_shp_similarity(det[2:4], fr_num)
                    gated = mot_sim*shp_sim < self.config.gating_thresh
                else:
                    raise NotImplementedError
                if not gated:
                    sim_mat[i, j] = mot_sim
                    num_matching_templates.append(len(prev_trk[j].historical_app) + 1)
                else:
                    num_matching_templates.append(0)
        return sim_mat, num_matching_templates

    def get_appearance_similarity(self, dets, prev_trk, sim_mat, num_matching_templates, bgr_img, fr_num):
        """
        Calculate appearance likelihoods
        :param dets: detections
        :param prev_trk: existing tracks
        :param sim_mat: current likelihood matrix
        :param num_matching_templates: list containing sequence length of each track
        :param bgr_img: bgr image
        :param fr_num: current frame number
        :return: appearance likelihood matrix
        """
        # Pre-calculate a Neural-Net input shape to a save time
        input_templates = np.zeros((sum(num_matching_templates), 128, 64, 6))
        input_shps = np.zeros((sum(num_matching_templates), 3))

        # Get detection templates
        det_templates = np.zeros((dets.shape[0], 128, 64, 3))
        for idx, det in enumerate(dets):
            cropped = bgr_img[det[1]:det[1] + det[3], det[0]:det[0] + det[2]]
            det_templates[idx, :, :, :] = cv2.resize(cropped, (64, 128))

        # Make input batch
        batch_sz = 0
        cur_idx = 0
        accum_num = 0
        for i in range(sim_mat.shape[0]):
            for j in range(sim_mat.shape[1]):
                if sim_mat[i, j] > 0:
                    matching_tmpls = []
                    matching_shps = []
                    anchor_shp = np.array([fr_num, *list(dets[i][2:4])], dtype=float)

                    # Historical appearances
                    for trk_tmpl, trk_shp, trk_fr in zip(prev_trk[j].historical_app, prev_trk[j].historical_shps,
                                                         prev_trk[j].historical_frs):
                        self.accumulate_app_pairs(matching_tmpls, trk_tmpl, det_templates[i])
                        self.accumulate_shp_pairs(matching_shps, trk_shp, trk_fr, anchor_shp)

                    # Recent appearance
                    self.accumulate_app_pairs(matching_tmpls, prev_trk[j].recent_app, det_templates[i])
                    self.accumulate_shp_pairs(matching_shps, prev_trk[j].recent_shp, prev_trk[j].recent_fr, anchor_shp)

                    input_templates[accum_num:accum_num + num_matching_templates[cur_idx], :, :, :] = np.array(
                        matching_tmpls)
                    input_shps[accum_num:accum_num + num_matching_templates[cur_idx], :] = np.array(matching_shps)
                    accum_num += num_matching_templates[cur_idx]
                    batch_sz += 1
                cur_idx += 1

        # JI-Net based matching-feature extraction
        feature_batch = self.NN.get_feature(input_templates)

        # Create LSTM input batch
        valid_matching_templates = [i for i in num_matching_templates if i > 0]

        input_batch = np.zeros((len(valid_matching_templates), self.config.lstm_len, 150 + 3))
        cur_idx = 0

        for idx, tmp_num in enumerate(valid_matching_templates):
            tmp_features = feature_batch[cur_idx:cur_idx + tmp_num]
            input_batch[idx, self.config.lstm_len - tmp_num:, :-3] = tmp_features
            input_batch[idx, self.config.lstm_len - tmp_num:, -3:] = input_shps[cur_idx:cur_idx + tmp_num]
            cur_idx += tmp_num

        # Final appearance likelihood from Deep-TAMA
        likelihoods = self.NN.get_likelihood(input_batch)

        return likelihoods

    def accumulate_shp_pairs(self, to_list, trk_shp, trk_fr, anchor_shp):
        """
        Accumulate pair of shape and frame
        :param to_list: accumulation list
        :param trk_shp: track shape
        :param trk_fr: Corresponding frame number of the track shape
        :param anchor_shp: anchor detection shape
        :return: None
        """
        trk_shp = np.array([trk_fr, *list(trk_shp)])
        matching_shp = trk_shp - anchor_shp
        matching_shp[0] /= self.fps
        matching_shp[1:3] /= anchor_shp[1:3]
        to_list.append(matching_shp)

    def accumulate_app_pairs(self, to_list, app_pair1, app_pair2):
        """
        Append pair of templates
        :param to_list: accumulation list
        :param app_pair1: template 1
        :param app_pair2: template 2
        :return: None
        """
        matching_tmpl = np.concatenate((normalization(app_pair1), normalization(app_pair2)), 2)
        to_list.append(matching_tmpl)

    def initialization(self, dets, bgr_img, fr_num):
        """
        Track initialization using MHT or delayed activation
        """
        if self.config.init_mode == 'mht':
            dets_unassoc = [True] * len(dets)
            if len(self.hyp_dets) == self.config.max_hyp_len:
                self.hyp_dets.pop(0)
                self.hyp_valid.pop(0)
                self.hyp_assoc.pop(0)
            if len(self.hyp_dets) > 0 and len(dets) > 0:
                prev_hyp = deepcopy(self.hyp_dets[-1])

                # Associate det and hyps
                tmp_hyp_dets, tmp_hyp_valid, tmp_hyp_assoc = self.init_association(dets, prev_hyp, dets_unassoc)
                self.hyp_dets.append(tmp_hyp_dets)
                self.hyp_valid.append(tmp_hyp_valid)
                self.hyp_assoc.append(tmp_hyp_assoc)

                # Init tracks from long hyps
                to_tracks = []
                if len(self.hyp_dets) == self.config.max_hyp_len:
                    for root_idx, _ in enumerate(self.hyp_assoc[-1]):
                        tmp_trk = [root_idx]  # Starting node
                        out_trk = self.recursive_find(self.config.max_hyp_len-1, root_idx, tmp_trk)
                        if out_trk:
                            tmp_to_track = []
                            for depth, hyp_idx in enumerate(out_trk):
                                self.hyp_valid[depth][hyp_idx] = False  # Mark used hyp idxs
                                tmp_to_track.append(self.hyp_dets[depth][hyp_idx])
                            to_tracks.append(tmp_to_track)

                # Recursively update trk
                if len(to_tracks) > 0:
                    for to_track in to_tracks:
                        concat_img = np.zeros((128, 0, 3), dtype=float)
                        tmp_trk = None
                        for i, y in enumerate(to_track):
                            tmp_fr = fr_num - self.config.max_hyp_len + i + 1
                            template = cv2.resize(self.imgs[i][y[1]:y[1] + y[3], y[0]:y[0] + y[2]], (64, 128))
                            concat_img = np.concatenate((concat_img, template), 1)
                            if i == 0:
                                tmp_trk = TrackState(y, template, tmp_fr, self.max_id, self.config)
                                self.max_id += 1
                            else:
                                tmp_trk.predict(tmp_fr, self.config)
                                tmp_trk.update(y, template, self.config.hist_thresh, tmp_fr, self.config, is_init=True)

                            # Initialization hypothesis restoration except a current state (semi online mode only)
                            if self.semi_on and i < self.config.max_hyp_len-1:
                                tmp_state = tmp_trk.save_info(tmp_fr)
                                self.trk_result[tmp_fr-1].append(tmp_state)

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

        elif self.config.init_mode == 'delay':
            # Delayed activation
            if len(self.hyp_state) > 0 and len(dets) > 0:
                dets = self.association(dets, self.hyp_state, bgr_img, fr_num)

                hyp_len = len(self.hyp_state)
                for i, hyp in enumerate(reversed(self.hyp_state)):
                    if hyp.recent_fr < fr_num:
                        # Remove unassociated hyps
                        self.hyp_state.pop(hyp_len - i - 1)
                    elif fr_num - hyp.init_fr + 1 == self.config.max_hyp_len:
                        # Promote unassociated hyps
                        hyp.track_id = self.max_id
                        self.max_id += 1
                        self.trk_state.append(hyp)

                        # Initialization hypothesis restoration except a current state (semi online mode only)
                        if self.semi_on:
                            for j in range(self.config.max_hyp_len - 1):
                                fr_offset = self.config.max_hyp_len - 1 - j
                                self.trk_result[fr_num - fr_offset - 1].append([hyp.track_id, *hyp.init_states[j], hyp.color])
                        self.hyp_state.pop(hyp_len - i - 1)
            # Init hyp
            for d in dets:
                template = cv2.resize(bgr_img[d[1]:d[1]+d[3], d[0]:d[0]+d[2]], (64, 128))
                tmp_trk = TrackState(d, template, fr_num, -1, self.config)
                self.hyp_state.append(tmp_trk)
            # Predict hyp next state
            for hyp in self.hyp_state:
                hyp.predict(fr_num + 1, self.config)
        else:
            raise NotImplementedError

    def init_association(self, dets, prev_hyp, dets_unassoc):
        """
        Associate remaining dets and potential tracks
        :param dets: dets
        :param prev_hyp: previous potential tracks
        :param dets_unassoc: boolean list
        :return: associated dets, validity of each hyp, associated hyps for each det
        """
        tmp_hyp_dets = []
        tmp_hyp_valid = []
        tmp_hyp_assoc = []
        for det_ind, det in enumerate(dets):
            is_assoc = False
            tmp_assoc = []
            tmp_conf = []

            # Hierarchical initialization
            for hyp_ind, hyp in enumerate(prev_hyp):
                # Strict matching
                iou_score = iou(det, hyp)

                if iou_score > self.config.assoc_iou_thresh:
                    is_assoc = True
                    tmp_assoc.append(hyp_ind)
                    tmp_conf.append(iou_score)
            if is_assoc:
                tmp_assoc = [_x for _, _x in sorted(zip(tmp_conf, tmp_assoc), key=lambda a: a[0], reverse=True)]
                dets_unassoc[det_ind] = False
                tmp_hyp_dets.append(det)
                tmp_hyp_valid.append(True)
                tmp_hyp_assoc.append(tmp_assoc)
            else:
                for hyp_ind, hyp in enumerate(prev_hyp):
                    # Weak matching
                    pos_dist, shp_sim = separate_measure(det, hyp)
                    if pos_dist < det[3] * self.config.assoc_dist_thresh and shp_sim > self.config.assoc_shp_thresh:
                        is_assoc = True
                        tmp_assoc.append(hyp_ind)
                        tmp_conf.append((-pos_dist) + shp_sim)
                if is_assoc:
                    tmp_assoc = [_x for _, _x in sorted(zip(tmp_conf, tmp_assoc), key=lambda a: a[0], reverse=True)]
                    dets_unassoc[det_ind] = False
                    tmp_hyp_dets.append(det)
                    tmp_hyp_valid.append(True)
                    tmp_hyp_assoc.append(tmp_assoc)

        return tmp_hyp_dets, tmp_hyp_valid, tmp_hyp_assoc

    def recursive_find(self, _depth, assoc_idx, _tmp_trk):
        """
        Recursively find valid track in hypothesis tree
        :param _depth: current tree depth
        :param assoc_idx: associated idx
        :param _tmp_trk: temporally created tracklet during tree traversal
        :return: valid track or empty list
        """
        if _depth == 0:
            return _tmp_trk

        for next_idx in self.hyp_assoc[_depth][assoc_idx]:
            if self.hyp_valid[_depth - 1][next_idx]:
                _tmp_trk.insert(0, next_idx)
                if self.recursive_find(_depth - 1, next_idx, _tmp_trk):
                    return _tmp_trk
                else:
                    _tmp_trk.pop(0)
        return []

    def track_interpolation(self, cur_fr):
        """
        Interpolate holes within an each track
        :param cur_fr: current frame number
        :return: None
        """
        cur_delay = min(self.fr_delay, cur_fr-1)
        for anchor_trk in self.trk_result[-1]:
            if cur_fr <= cur_delay:
                continue

            # Find same id in fr_delay range
            found = False
            found_i = 0
            found_trk = None
            i = 1
            while not found and i <= cur_delay:
                for refer_trk in self.trk_result[-i-1]:
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
        """
        Append valid track states to the result list
        :param fr_num: frame number
        :return: None
        """
        tmp_save = []
        valid_miss_num = 1
        if len(self.trk_state) > 0:
            for trk in self.trk_state:
                if trk.recent_fr > fr_num - valid_miss_num:
                    tmp_state = trk.save_info(fr_num)
                    tmp_save.append(tmp_state)

        self.trk_result[fr_num-1].extend(tmp_save)
