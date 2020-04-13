import numpy as np
import random
# from scipy.stats import multivariate_normal


class Track:
    """
    Track with Kalman filtering
    """
    def __init__(self, init_y, init_app, init_fr, track_id):

        # init_y : [x, y, w, h]
        self.X = np.transpose(np.array([init_y[0],  0, init_y[1], 0]))

        # state transition matrix
        self.Ts = 1 # frame rates
        self.F1 = np.array([1, self.Ts], [0, 1])
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

        self.historical_frs = []
        self.recent_fr = init_fr
        self.shapes = []
        self.recent_app = init_app  # recent appearance which may be unreliable
        self.historical_app = []  # set of appearances which are reliable
        self.shape = init_y[2:4]  # width and height
        self.color = [random.randrange(256), random.randrange(256), random.randrange(256)]
        self.track_id = track_id

    def predict(self):
        self.X = self.F @ self.X
        self.P = self.F @ self.P @ np.transpose(self.F) + self.Q

        return self.X, self.P

    def update(self, y):
        IM = self.H @ self.X
        IS = self.R + self.H @ self.P @ self.H.T
        K = (self.P @ self.H.T)/IS
        self.X = self.X + K @ (y-IM)
        self.P = self.P - K @ IS @ K.T

        return self.X, self.P

    def mahalanobis_distance(self, y):
        X = np.array([self.X[0], self.X[2]]).T
        d_squared = np.exp(-0.5 * (X-y).T @ np.linalg.inv(self.pos_var) @ (X-y))

        return d_squared