import os
import numpy as np
from scipy.optimize import linear_sum_assignment
import dataLoader as ds
from copy import deepcopy
import cv2
from kalmanFilter import TrackState
from Tools import IOU, nms

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

    def initTrk(self, init_inds):
        for hyp_ind in init_inds:
            tmp_trk = [self.max_id]
            for hyp_bb in self.trk_hyp[hyp_ind]:
                tmp_trk.append(hyp_bb)
            self.trk_state.append(tmp_trk)
            self.max_id += 1

        self.delHyp(init_inds)

    def updateTrk(self, updated_trk):
        self.trk_state = deepcopy(updated_trk)

        return self.trk_state

    def terminateTrk(self, del_inds):
        for ind in del_inds:
            self.trk_state.pop(ind)

        return self.trk_state

    def addHyp(self, det, cur_fr):
        self.trk_hyp.append([list(np.append(det, cur_fr))])

        return self.trk_hyp

    def updateHyp(self, updated_hyp):
        self.trk_hyp = deepcopy(updated_hyp)

        return self.trk_hyp

    def delHyp(self, del_inds):
        for hyp_ind in del_inds:
            self.trk_hyp.pop(hyp_ind)

        return self.trk_hyp

    def visualizeTrk(self, img):
        return NotImplementedError

def startTrack():
    track = Track()
    data = ds.data(is_test=True)
    frame_end = 1000
    assoc_thresh = 0.5
    max_state_num = 5
    max_hyp_len = 4
    det_thresh = 0.2

    seq_name = "MOT16-02"

    # get sequence info
    seq_info = data.get_seq_info(seq_name=seq_name)

    # 1st frame exception
    bgr_img, dets = data.get_frame_info(seq_name=seq_name, frame_num=1)
    dets = dets[dets[:, 4] > det_thresh].astype(np.int)
    for det in dets:
        track.addHyp(det[1:5], 1)

    for fr_num in range(2, frame_end+1):
        bgr_img, dets = data.get_frame_info(seq_name="MOT16-02", frame_num=fr_num)
        dets = dets[dets[:, 5] > det_thresh].astype(int)

        # Track-detection association
        prev_trk = deepcopy(track.trk_state)

        if len(prev_trk) > 0 and len(dets) > 0:

            sim_mat = np.zeros((len(dets), len(prev_trk)))

            # Remove last void track column
            row_ind, col_ind = linear_sum_assignment(-sim_mat)

            # Track update
            assoc_det_ind = []
            for trk_ind, det_ind in zip(col_ind, row_ind):
                if sim_mat[det_ind, trk_ind] > assoc_thresh:
                    assoc_det_ind.append(det_ind)

            # Associated det removal
            dets = np.delete(dets, assoc_det_ind, axis=0)
            print('remaining det : {}'.format(len(dets)))

            # Track termination

        # Track initialization
        prev_hyp = deepcopy(track.trk_hyp)
        unassoc_hyps = [i for i in range(0, len(prev_hyp))]
        unassoc_dets = [i for i in range(0, len(dets))]

        tmp_hyp_dets = []
        tmp_hyp_valid = []
        tmp_hyp_assoc = []
        if len(dets) > 0 and len(prev_hyp) > 0:
            for det_ind in unassoc_dets:
                det = dets[det_ind]
                is_assoc = False
                tmp_assoc = []
                for hyp_ind in unassoc_hyps:
                    hyp = track.trk_hyp[hyp_ind]
                    if IOU(det, hyp) > 0.4:
                        is_assoc = True
                        tmp_assoc.append(hyp_ind)
                if is_assoc:
                    tmp_hyp_dets.append(det)
                    tmp_hyp_valid.append(True)
                    tmp_hyp_assoc.append(tmp_assoc)

        if len(track.hyp_dets) > max_hyp_len:
            track.hyp_dets.pop(0)
            track.hyp_valid.pop(0)
            track.hyp_assoc.pop(0)

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
                if track.hyp_valid[_depth-1][next_idx] & track.hyp_assoc[_depth-1][next_idx]:
                    if recursive_find(_depth-1, next_idx, _tmp_trk.insert(0, next_idx)):
                        return _tmp_trk
                    else:
                        _tmp_trk.pop(0)
            return []

        # Init trk
        to_tracks = []
        if len(track.hyp_dets) > max_hyp_len:
            for tail_assoc in track.hyp_assoc[-1]:
                for tail_assoc_idx in tail_assoc:
                    tmp_trk = [tail_assoc_idx]
                    if recursive_find(max_hyp_len-1, tail_assoc_idx, tmp_trk):
                        tmp_to_track = []
                        for depth, hyp_idx in enumerate(tmp_trk):
                            track.hyp_valid[depth][hyp_idx] = False
                            tmp_to_track.insert(0, track.hyp_dets[depth][hyp_idx])
                        to_tracks.append(tmp_to_track)

        for to_track in to_tracks:
            tmp_y = to_track[0]
            init_fr = fr_num
            tmp_state = TrackState()
            track.trk_state.append(tmp_state)

            track.addHyp(det[1:5], fr_num)

        # Init hyp
        for det_ind in unassoc_dets:
            track.hyp_dets.append(dets[det_ind])
            track.hyp_valid.append(True)
            track.hyp_assoc.append([])

        # Track visualization
        if len(track.trk_state) > 0:
            for trk in track.trk_state:
                if trk[-1][-1] == fr_num:
                    cv2.putText(bgr_img, str(trk[0]), tuple(trk[-1][0:2]), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
                    cv2.rectangle(bgr_img, tuple(trk[-1][0:2]), tuple([trk[-1][0]+trk[-1][2], trk[-1][1]+trk[-1][3]]), (0, 0, 255), 3)
            cv2.imshow('{}'.format(fr_num), bgr_img)
            cv2.waitKey(-1)
            cv2.destroyAllWindows()

    return NotImplementedError


if __name__=="__main__":
    startTrack()