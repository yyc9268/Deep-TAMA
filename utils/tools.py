import copy
import cv2
import math
import numpy as np
"""
Collection of tools handling bounding-boxes
"""


def det_preprocessing(dets, fr_num, nms_iou_thresh, det_thresh):
    """
    Detection pre-processing (negative value refinement, NMS, thresholding)
    :param dets: [[-1, x, y, w, h, conf], ...]
    :param fr_num: frame number
    :return: pre-processed detections
    """
    if len(dets) > 0:
        dets = dets[:, 1:6]
        for i in range(dets.shape[0]):
            dets[i][dets[i] < 0] = 0
        dets = np.hstack((dets, np.ones((dets.shape[0], 1)) * fr_num))  # [x, y, w, h, conf, fr]
        keep = nms(dets, nms_iou_thresh)
        dets = dets[keep]
        dets = dets[dets[:, 4] > det_thresh].astype(int)
    else:
        dets = []

    return dets


def nms(orig_dets, iou_thresh=0.5):
    """
    Non-Maximum-Suppression
    :param orig_dets: [[x, y, w, h, conf, fr], ..., [x, y, w, h, conf, fr]]
    :param iou_thresh: IoU threshold between dets
    :return: keeping det indices
    """
    dets = copy.deepcopy(orig_dets)
    dets = dets[dets[:, 4].argsort()[::-1]]
    keep = [True]*len(dets)
    for i in range(0, len(dets)-1):
        for j in range(i+1, len(dets)):
            if iou(dets[i], dets[j]) > iou_thresh:
                keep[j] = False
    return keep


def iou(bb1, bb2):
    """
    Intersection over union
    :param bb1: [x, y, w, h]
    :param bb2: [x, y, w, h]
    :return: IoU between bb1 and bb2
    """
    x1 = max(bb1[0], bb2[0])
    y1 = max(bb1[1], bb2[1])
    x2 = min(bb1[0]+bb1[2], bb2[0]+bb2[2])
    y2 = min(bb1[1]+bb1[3], bb2[1]+bb2[3])

    intersection = max(0, x2-x1+1)*max(0, y2-y1+1)

    bb1_area = bb1[2]*bb1[3]
    bb2_area = bb2[2]*bb2[3]

    iou = intersection/float(bb1_area + bb2_area - intersection)
    return iou


def separate_measure(bb1, bb2):
    """
    Get positional distance and shape similarity
    :param bb1: [x, y, w, h]
    :param bb2: [x, y, w, h]
    :return: positional distance, shape similarity
    """
    cpos1 = [bb1[0]+bb1[2]/2, bb1[1]+bb1[3]/2]
    cpos2 = [bb2[0]+bb2[2]/2, bb2[1]+bb2[3]/2]

    pos_dist = math.sqrt((cpos1[0]-cpos2[0])**2 + (cpos1[1]-cpos2[1])**2)
    shp_dist = min(bb1[3]/bb2[3], bb2[3]/bb1[3])

    return pos_dist, shp_dist


def normalization(img):
    """
    :param img: numpy array image
    :return: normalized image in which values are in range [0, 1]
    """
    norm_img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return norm_img


def denormalization(img: np.array) -> np.array:
    """
    Warning : this is only for debugging
    :param img: numpy array image
    :return: denormalized image in which values are in range [0, 255]
    """
    denorm_img = img * 255
    return denorm_img


def augment_bbox(bbox, very_noisy=False):
    """
    Add gaussian noise on center location & width and height
    - center noise += width/height * N(0, 0.1)
    - width/height *= N(1, 0.1)
    :param bbox: [x, y, w, h]
    :return: augmented bounding-box
    """
    if very_noisy:
        loc_aug_ratio = np.random.normal(0, 0.1)
        wh_aug_ratio = np.random.normal(1, 0.2)
    else:
        loc_aug_ratio = np.random.normal(0, 0.05)
        wh_aug_ratio = np.random.normal(1, 0.1)

    augmented_bbox = copy.deepcopy(bbox)
    augmented_bbox[0:2] += augmented_bbox[3:4] * loc_aug_ratio
    augmented_bbox[2:4] *= wh_aug_ratio

    return augmented_bbox
