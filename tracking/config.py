
class config:
    def __init__(self, fps):
        self.fps = fps

        # Tracking parameters
        self.det_thresh = 0.1
        self.nms_iou_thresh = 0.5
        self.assoc_iou_thresh = 0.45 * (1.0 / (1.0 + 1.0 * max(0.0, min(0.5, 1.0/self.fps))))
        self.assoc_shp_thresh = 0.8 * (1.0 / (1.0 + 1.0 * max(0.0, min(0.5, 1.0/self.fps))))
        self.assoc_dist_thresh = 0.5 * ((1.0 + max(0.0, 1.0 * min(0.5, 1.0 / self.fps))) / 1.0)
        self.max_hyp_len = 4
        self.miss_thresh = 3 * self.fps
        self.assoc_thresh = 0.4
        self.gating_thresh = self.assoc_thresh + 0.2
        self.init_conf = 0.7

        # Historical appearance management parameters
        self.hist_thresh = 0.7
        self.max_hist_len = 5
        self.lstm_len = self.max_hist_len+1
        self.min_hist_intv = 0.2 * self.fps
        self.max_hist_age = 2 * self.fps