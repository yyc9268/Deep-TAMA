import os


class Config:
    def __init__(self, fps=30):
        # Change this path to the users own dataset path
        self.desktop_path = os.path.expanduser("~\Desktop")
        self.seq_path = os.path.join(self.desktop_path, "dataset", 'MOT')

        # Tracking parameters
        self.det_thresh = 0.1
        self.nms_iou_thresh = 0.4
        self.max_hyp_len = 5
        self.alpha = 1
        self.valid_miss_frame = 0  # maximum consecutive frames with only predictions to be included in results

        self.semi_on = False
        self.use_appearance = True
        self.init_mode = 'mht'  # mht or delay
        self.gating_mode = "iou"  # iou or mahalanobis

        if self.gating_mode == "iou":  # Good for general purposes
            self.gating_thresh = 0.3
            self.assoc_thresh = self.gating_thresh
        elif self.gating_mode == "maha":  # Good for top-view perspective
            self.gating_thresh = 0.5
            self.assoc_thresh = self.gating_thresh
        else:
            raise NotImplementedError

        # Historical appearance management parameters
        self.hist_thresh = self.gating_thresh + 0.2
        self.init_conf = self.hist_thresh
        self.max_hist_len = 5
        self.lstm_len = self.max_hist_len+1

        self.fps = fps

        # Training parameters
        self.log_dir = 'log'
        self.model_dir = 'model'
        self.np_val_dir = 'utils'

        self.model = {'jinet': {}, 'lstm': {}}
        model_name = 'jinet'
        self.model[model_name]['init_lr'] = 1e-2
        self.model[model_name]['epoch_batch_len'] = 1024
        self.model[model_name]['train_batch_len'] = 32
        self.model[model_name]['val_batch_len'] = 32
        self.model[model_name]['tot_epoch'] = 2000
        self.model[model_name]['repeat'] = 2  # Repeat training with same epoch data without decaying learning rate
        self.model[model_name]['log_intv'] = 20
        self.model[model_name]['save_name'] = 'JINet-model'
        model_name = 'lstm'
        self.model[model_name]['init_lr'] = 1e-2
        self.model[model_name]['epoch_batch_len'] = 1024
        self.model[model_name]['train_batch_len'] = 32
        self.model[model_name]['val_batch_len'] = 32
        self.model[model_name]['tot_epoch'] = 2000
        self.model[model_name]['repeat'] = 2  # Repeat training with same epoch data without decaying learning rate
        self.model[model_name]['log_intv'] = 20
        self.model[model_name]['save_name'] = 'DeepTAMA-model'

    @property
    def fps(self):
        return self._fps

    @fps.setter
    def fps(self, fps):
        self._fps = fps
        self.calc_fps_parameters()

    @property
    def det_thresh(self):
        return self._det_thresh

    @det_thresh.setter
    def det_thresh(self, det_thresh):
        self._det_thresh = det_thresh

    def calc_fps_parameters(self):
        # print("Recalculate thresholds with FPS : {}".format(self.fps))
        self.assoc_iou_thresh = 0.45 * (1.0 / (1.0 + 1.0 * max(0.0, min(0.5, 1.0 / self.fps))))
        self.assoc_shp_thresh = 0.8 * (1.0 / (1.0 + 1.0 * max(0.0, min(0.5, 1.0 / self.fps))))
        self.assoc_dist_thresh = 0.3 * ((1.0 + max(0.0, 1.0 * min(0.5, 1.0 / self.fps))) / 1.0)
        self.miss_thresh = self.alpha * self.fps
        self.min_hist_intv = 0.15 * self.fps
        self.max_hist_age = self.alpha * self.fps
