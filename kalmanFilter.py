import numpy as np
import random
from Config import Config
# from scipy.stats import multivariate_normal


class TrackState:
    """
    Track with Kalman filtering
    """
    def __init__(self, init_y, init_app, init_fr, track_id):

        # init_y : [x, y, w, h, conf, fr]
        self.X = np.array([[init_y[0]],  [0], [init_y[1]], [0]])

        # state transition matrix
        self.Ts = 1 # frame rates
        self.F1 = np.array([[1, self.Ts], [0, 1]])
        self.Fz = np.zeros((2,2))
        self.F = np.concatenate((np.concatenate((self.F1, self.Fz), 1),
                                 np.concatenate((self.Fz, self.F1), 1)), 0)

        # dynamic model covariance
        self.q = 0.05
        self.Q1 = np.array([[self.Ts**4, self.Ts**2], [self.Ts**2, self.Ts]])*(self.q**2)
        self.Q = np.concatenate((np.concatenate((self.Q1, self.Fz), 1),
                                 np.concatenate((self.Fz, self.Q1), 1)), 0)

        # Initial error covariance
        self.ppp = 5
        self.P = np.eye(4)*self.ppp
        self.H = np.array([[1,0,0,0], [0,0,1,0]])  # H matrix: measurement model
        self.R = 0.1*np.eye(2)  # measurement model covariance

        # covariance used for motion affinity evaluation
        self.pos_var = np.diag([30**2, 75**2])

        self.recent_app = init_app  # recent appearance which may be unreliable
        self.recent_conf = 0.5
        self.recent_fr = init_fr
        self.recent_shp = init_y[2:4]  # width and height

        self.historical_app = []  # set of appearances which are reliable
        self.historical_conf = []
        self.historical_frs = []
        self.historical_shps = []

        self.color = (random.randrange(256), random.randrange(256), random.randrange(256))
        self.track_id = track_id

    def predict(self, param, fr):
        self.X = self.F @ self.X
        self.P = self.F @ self.P @ np.transpose(self.F) + self.Q

        # Historical Appearance Management
        # Deletion rule
        if fr - self.historical_frs[0] > param.max_hist_age:
            self.pop_first_hist_element()

        return self.X, self.P

    def update(self, y, app, param, fr):
        Y = np.array([[y[0]], [y[1]]])
        IM = self.H @ self.X
        IS = self.R + self.H @ self.P @ self.H.T
        K = (self.P @ self.H.T)@np.linalg.inv(IS)

        self.X = self.X + K @ (Y-IM)
        self.P = self.P - K @ IS @ K.T

        # Historical Appearance Management
        # Addition rule
        if self.recent_conf > param.hist_thresh:
            self.add_recent_to_hist(param, fr)

        # Deletion rule
        if len(self.historical_app) > param.max_hist_len:
            self.pop_first_hist_element()

        self.recent_app = app
        self.recent_fr = fr
        self.recent_shp = y[2:4]

        return self.X, self.P

    def pop_first_hist_element(self):
        self.historical_app.pop(0)
        self.historical_conf.pop(0)
        self.historical_frs.pop(0)
        self.historical_frs.pop(0)

    def add_recent_to_hist(self, param, fr):
        if fr - self.recent_fr >= param.min_hist_intv:
            self.historical_app.append(self.recent_app)
            self.historical_conf.append(self.recent_conf)
            self.historical_frs.append(self.recent_fr)
            self.historical_shps.append(self.recent_shp)

    def mahalanobis_distance(self, y):
        X = np.array([[self.X[0][0], self.X[2][0]]])
        Y = np.array([[y[0], y[1]]])
        d_squared = np.exp(-0.5 * (X-Y) @ np.linalg.inv(self.pos_var) @ (X-Y).T)

        return d_squared[0][0]