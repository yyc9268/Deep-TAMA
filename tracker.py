import os
import numpy as np
from scipy.optimize import linear_sum_assignment
import dataLoader as ds
from copy import deepcopy
import cv2
from kalmanFilter import TrackState
from Tools import IOU, NMS
from Config import Config

desktop_path = os.path.expanduser("~\Desktop")
data_path = os.path.join(desktop_path, "dataset")
seq_path = os.path.join(data_path, "MOT16", "train")
seqs = np.array(os.listdir(seq_path))

class Track:
    def __init__(self):
        """
        Track state : [id, [x, y, w, h, fr], ..., [x, y, w, h, fr]]
        Hypothesis state : [[x, y, w, h, fr], ... [x, y, w, h, fr]]
        """
        self.trk_result = []
        self.trk_state = []
        self.hyp_dets = []  # [[], [], [], [], []]
        self.hyp_valid = [] # [[1,0], [1,1], ..., [1,1,1]]
        self.hyp_assoc = []  # [[0, 0, 0], [[1, 3], 2, 2], [], [], []]
        self.max_id = 1


def startTrack():
    track = Track()
    data = ds.data(is_test=True)

    frame_end = 1000

    seq_name = "MOT16-02"
    if not os.path.exists(seq_name):
        os.mkdir(seq_name)

    # get sequence info
    seq_info = data.get_seq_info(seq_name=seq_name)

    # get tracking parameters
    Conf = Config(seq_info[-1])

    # 1st frame exception
    """
    bgr_img, dets = data.get_frame_info(seq_name=seq_name, frame_num=1)
    dets = dets[:, 1:6]
    dets = np.hstack((dets, np.ones((dets.shape[0], 1))))
    keep = NMS(dets, iou_thresh)
    dets = dets[keep]

    dets = dets[dets[:, 4] > det_thresh].astype(np.int)
    track.hyp_dets.append([])
    track.hyp_valid.append([])
    track.hyp_assoc.append([])
    for unassoc_det in dets:
        track.hyp_dets[-1].append(unassoc_det)
        track.hyp_valid[-1].append(True)
        track.hyp_assoc[-1].append([])
    """

    for fr_num in range(1, frame_end+1):
        bgr_img, dets = data.get_frame_info(seq_name="MOT16-02", frame_num=fr_num)
        dets = dets[:, 1:6]
        dets = np.hstack((dets, np.ones((dets.shape[0], 1))*fr_num))
        keep = NMS(dets, Conf.nms_iou_thresh)
        dets = dets[keep]
        # dets : [[x,y,w,h,conf], ..., [x,y,w,h,conf]]
        dets = dets[dets[:, 5] > Conf.det_thresh].astype(int)

        # Track-detection association
        prev_trk = deepcopy(track.trk_state)

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
                if sim_mat[det_ind, trk_ind] > Conf.assoc_thresh:
                    track.trk_state[trk_ind].update(dets[det_ind], fr_num)
                    assoc_det_ind.append(det_ind)

            # Associated det removal
            dets = np.delete(dets, assoc_det_ind, axis=0)

        # Track initialization
        dets_unassoc = [True] * len(dets)
        if len(track.hyp_dets) > Conf.max_hyp_len:
            track.hyp_dets.pop(0)
            track.hyp_valid.pop(0)
            track.hyp_assoc.pop(0)

        if len(track.hyp_dets) > 0 and len(dets) > 0:
            prev_hyp = deepcopy(track.hyp_dets[-1])

            tmp_hyp_dets = []
            tmp_hyp_valid = []
            tmp_hyp_assoc = []
            if len(dets) > 0 and len(prev_hyp) > 0:
                for det_ind, det in enumerate(dets):
                    is_assoc = False
                    tmp_assoc = []
                    for hyp_ind, hyp in enumerate(prev_hyp):
                        # Hierarchical initialization
                        if IOU(det, hyp) > Conf.assoc_iou_thresh:
                            is_assoc = True
                            tmp_assoc.append(hyp_ind)
                    if is_assoc:
                        dets_unassoc[det_ind] = False
                        tmp_hyp_dets.append(det)
                        tmp_hyp_valid.append(True)
                        tmp_hyp_assoc.append(tmp_assoc)

            track.hyp_dets.append(tmp_hyp_dets)
            track.hyp_valid.append(tmp_hyp_valid)
            track.hyp_assoc.append(tmp_hyp_assoc)

            def recursive_find(_depth, assoc_idx, _tmp_trk):
                if _depth == 0:
                    if track.hyp_valid[_depth][assoc_idx]:
                        return _tmp_trk
                    else:
                        return []

                for next_idx in track.hyp_assoc[_depth][assoc_idx]:
                    if track.hyp_valid[_depth-1][next_idx] and track.hyp_assoc[_depth-1][next_idx]:
                        _tmp_trk.insert(0, next_idx)
                        if recursive_find(_depth-1, next_idx, _tmp_trk):
                            return _tmp_trk
                        else:
                            _tmp_trk.pop(0)
                return []

            # Init trk
            to_tracks = []
            if len(track.hyp_dets) > Conf.max_hyp_len:
                for tail_assoc in track.hyp_assoc[-1]:
                    for tail_assoc_idx in tail_assoc:
                        tmp_trk = [tail_assoc_idx]
                        out_trk = recursive_find(Conf.max_hyp_len-1, tail_assoc_idx, tmp_trk)
                        if out_trk:
                            tmp_to_track = []
                            for depth, hyp_idx in enumerate(out_trk):
                                track.hyp_valid[depth][hyp_idx] = False
                                tmp_to_track.insert(0, track.hyp_dets[depth][hyp_idx])
                            to_tracks.append(tmp_to_track)

            # Recursively update trk
            for to_track in to_tracks:
                for i, y in enumerate(to_track):
                    tmp_fr = fr_num - Conf.max_hyp_len + i
                    template = bgr_img[y[1]:y[1] + y[3], y[0]:y[0] + y[2]]
                    if i == 0:
                        tmp_state = TrackState(y, tmp_fr, template, track.max_id)
                        track.max_id += 1
                    else:
                        tmp_state.predict()
                        tmp_state.update(y, tmp_fr)

                track.trk_state.append(tmp_state)

        else:
            track.hyp_dets.append([])
            track.hyp_valid.append([])
            track.hyp_assoc.append([])

        # Init hyp
        unassoc_dets = dets[dets_unassoc]
        for unassoc_det in unassoc_dets:
            track.hyp_dets[-1].append(unassoc_det)
            track.hyp_valid[-1].append(True)
            track.hyp_assoc[-1].append([])

        # Track visualization
        valid_miss_num = 5
        if len(track.trk_state) > 0:
            for trk in track.trk_state:
                if trk.recent_fr > fr_num-valid_miss_num:
                    tmp_X = list(map(int, trk.X))
                    cv2.putText(bgr_img, str(trk.track_id), (tmp_X[0], tmp_X[2]), cv2.FONT_HERSHEY_COMPLEX, 2, trk.color, 2)
                    cv2.rectangle(bgr_img, (tmp_X[0], tmp_X[2]), (tmp_X[0]+trk.shape[0], tmp_X[2]+trk.shape[1]), trk.color, 3)

            cv2.imwrite(os.path.join(seq_name, "{}.jpg".format(fr_num)), bgr_img)
            #cv2.imshow('{}'.format(fr_num), bgr_img)
            #cv2.waitKey(-1)
            #cv2.destroyAllWindows()

        # Track termination
        prev_trk_len = len(track.trk_state)
        for i, trk in enumerate(track.trk_state[::-1]):
            if fr_num - trk.recent_fr > Conf.miss_thresh:
                track.trk_state.pop(prev_trk_len - i - 1)
            elif trk.X[0, 0] <= 0 or trk.X[2, 0] <= 0 or trk.X[0, 0] >= seq_info[0] or trk.X[2, 0] >= seq_info[1]:
                track.trk_state.pop(prev_trk_len - i - 1)

        # Next frame prediction
        for trk in track.trk_state:
            trk.predict()

    return NotImplementedError


if __name__=="__main__":
    startTrack()