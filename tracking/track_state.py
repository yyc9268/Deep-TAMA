import numpy as np
import random


chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class TrackState:
    """
    Track with Kalman filtering & Historical appearance management
    """
    def __init__(self, init_y, init_app, init_fr, track_id, param):
        """
        Initialize track state
        State is a center of a bounding-box
        :param init_y: [x, y, w, h, conf, fr]
        :param init_app: image template
        :param init_fr: frame number
        :param track_id: current track id
        :param param: tracking parameters
        """
        # LT to center
        self.X = np.array([[init_y[0]+init_y[2]/2],  [0], [init_y[1]+init_y[3]/2], [0]])

        # state transition matrix
        self.Ts = 1
        self.F1 = np.array([[1, self.Ts], [0, 1]])
        self.Fz = np.zeros((2, 2))
        self.F = np.concatenate((np.concatenate((self.F1, self.Fz), 1),
                                 np.concatenate((self.Fz, self.F1), 1)), 0)

        # dynamic model covariance
        self.q = 0.05
        self.Q1 = np.array([[self.Ts**4, self.Ts**2], [self.Ts**2, self.Ts]])*(self.q**2)
        self.Q = np.concatenate((np.concatenate((self.Q1, self.Fz), 1),
                                 np.concatenate((self.Fz, self.Q1), 1)), 0)

        # Initial error covariance
        self.ppp = 100
        self.P = np.eye(4)*self.ppp
        self.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])  # H matrix: measurement model
        self.R = 10*np.eye(2)  # measurement model covariance, we assume 10 pixels

        # covariance used for motion affinity evaluation
        self.pos_var = np.diag([30**2, 75**2])

        self.recent_app = init_app  # recent appearance which may be unreliable
        self.recent_conf = param.hist_thresh + 0.1
        self.recent_fr = init_fr
        self.init_fr = init_fr
        self.recent_shp = init_y[2:4]  # width and height

        self.historical_app = []  # set of appearances which are reliable
        self.historical_conf = []
        self.historical_frs = []
        self.historical_shps = []

        self.init_states = []  # For initial N-frame restoration

        self.color = (random.randrange(256), random.randrange(256), random.randrange(256))
        self.track_id = track_id

    def predict(self, fr, param):
        """
        Predict track state
        :param fr: current frame number
        :param param: track management parameters
        :return: predicted state, covariance
        """
        if self.track_id < 0:  # Save states before activated
            self.init_states.append(self.bbox_info(fr-1))

        self.X = self.F @ self.X
        self.P = self.F @ self.P @ np.transpose(self.F) + self.Q

        # Historical Appearance Management
        # Deletion rule
        if len(self.historical_app) > 0:
            if fr - self.historical_frs[0] > param.max_hist_age:
                self.pop_first_hist_element()

        return self.X, self.P

    def update(self, y, app, conf, fr, param, is_init=False):
        """
        Update track state
        :param y: observation
        :param app: image template
        :param conf: confidence
        :param fr: current frame number
        :param param: update parameters
        :param is_init: initalization mode
        :return: updated track state, covariance
        """
        # LT to center
        Y = np.array([[y[0]+y[2]/2], [y[1]+y[3]/2]])
        IM = self.H @ self.X
        IS = self.R + self.H @ self.P @ self.H.T
        K = (self.P @ self.H.T)@np.linalg.inv(IS)

        self.X = self.X + K @ (Y-IM)
        self.P = self.P - K @ IS @ K.T

        # Historical Appearance Management
        # Addition rule
        if self.recent_conf > param.hist_thresh or is_init:
            self.add_recent_to_hist(param, is_init)

        # Deletion rule
        if len(self.historical_app) > param.max_hist_len:
            self.pop_first_hist_element()

        self.recent_app = app
        self.recent_conf = conf
        self.recent_fr = fr
        self.recent_shp = y[2:4]

        return self.X, self.P

    def pop_first_hist_element(self):
        """
        Pop first element from historical appearance queue
        """
        self.historical_app.pop(0)
        self.historical_conf.pop(0)
        self.historical_frs.pop(0)
        self.historical_shps.pop(0)

    def add_recent_to_hist(self, param, is_init):
        """
        Append new element to historical appearance queue
        :param param: tracking parameters
        :param is_init: initialization mode
        :return: None
        """
        if len(self.historical_app) == 0 or is_init or (self.recent_fr - self.historical_frs[-1]) >= param.min_hist_intv:
            self.historical_app.append(self.recent_app)
            self.historical_conf.append(self.recent_conf)
            self.historical_frs.append(self.recent_fr)
            self.historical_shps.append(self.recent_shp)

    def bbox_info(self, fr):
        """
        Return bounding-box information of current frame
        :param fr: current frame
        :return: [left, top, width, height]
        """
        shp = self.get_shp(fr)
        bbox_info = [self.X[0][0]-shp[0]/2, self.X[2][0]-shp[1]/2, shp[0], shp[1]]

        return bbox_info

    def save_info(self, fr):
        """
        Return track state to save
        :param fr: current frame number
        :return: [track id, [left, top, width, height], color]
        """
        bbox = self.bbox_info(fr)
        save_state = [self.track_id, *bbox, self.color]

        return save_state

    def get_shp(self, fr):
        # Get predicted shape
        if len(self.historical_shps) == 0:
            pred_shp = self.recent_shp
        else:
            intp_fr = fr - self.recent_fr
            tmp_pred_shp = (lambda a, b, fr_diff: [a[0] + intp_fr*(a[0]-b[0])/fr_diff, a[1] + intp_fr*(a[1]-b[1])/fr_diff])\
                (self.recent_shp, self.historical_shps[-1], self.recent_fr - self.historical_frs[-1])
            pred_shp = (self.recent_shp + tmp_pred_shp)/2  # Blend recent shape and confident shape

        return pred_shp

    def get_shp_similarity(self, y, fr):
        """
        Calculate a shape a similarity with an observation
        :param y: observation
        :param fr: current frame
        :return: shape similarity
        """
        pred_shp = self.get_shp(fr)
        shp_sim = (min(pred_shp[0] / y[0], y[0] / pred_shp[0]) + min(pred_shp[1] / y[1], y[1] / pred_shp[1]))/2
        # shp_sim = np.exp(-1.5 * (abs(y[0]-pred_shp[0])/(y[0]+pred_shp[0]) + abs(y[1]-pred_shp[1])/(y[1]+pred_shp[1])))

        return shp_sim

    def mahalanobis_similarity(self, y):
        """
        Calculate pseudo mahalanobis distance-based likelihood (constant covariance matrix)
        :param y: observation
        :return: mahalanobis_distance-based likelihood
        """
        X = np.array([[self.X[0][0], self.X[2][0]]])
        Y = np.array([[y[0]+y[2]/2, y[1]+y[3]/2]])
        d_squared = np.exp(-0.5 * (X-Y) @ np.linalg.inv(self.pos_var) @ (X-Y).T)

        return d_squared[0][0]
