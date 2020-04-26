
class Config:
    def __init__(self, fps):
        self.fps = fps

        # Tracking parameters
        self.det_thresh = 0.2
        self.nms_iou_thresh = 0.4
        self.assoc_iou_thresh = 0.45 * (1.0 / (1.0 + max(0.0, min(0.5, 1.0/self.fps))))
        self.assoc_dist_thresh = 0.8 * (1.0 / (1.0 + max(0.0, min(0.5, 1.0/self.fps))))
        self.max_hyp_len = 4
        self.miss_thresh = self.fps
        self.assoc_thresh = 0.4
        self.init_conf = 0.6

        # Historical appearance management parameters
        self.hist_thresh = 0.6
        self.max_hist_len = 14
        self.min_hist_intv = 0.2 * self.fps
        self.max_hist_age = 2 * self.fps