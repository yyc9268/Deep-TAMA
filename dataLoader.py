import numpy as np
import os
import random
import cv2
import operator
from copy import deepcopy

desktop_path = os.path.expanduser("~\Desktop")
seq_path = os.path.join(desktop_path, "dataset", 'MOT')
#seq_path = os.path.join(data_path, "2DMOT2015", "train")
#seq_path = os.path.join(data_path, "CVPR19Labels", "train")
all_seqs = np.array(os.listdir(seq_path))
seqs = []

training_set = ['TUD-Campus', 'ETH-Sunnyday', 'KITTI-17', 'MOT16-04', 'MOT16-05', 'MOT16-09', 'MOT16-10', 'MOT16-11', 'MOT16-13']
validation_set = ['TUD-Stadtmitte', 'PETS09-S2L1', 'ETH-Bahnhof', 'KITTI-13', 'MOT16-02']

# Sequence info
# [width, height, fps]
training_info = [[640, 480, 25], [640, 480, 14], [1224, 370, 10], [1920, 1080, 30], [640, 480, 14], [1920, 1080, 30], [1920, 1080, 30], [1920, 1080, 30], [1920, 1080, 25]]
validation_info = [[640, 480, 25], [768, 576, 7], [640, 480, 14], [1242, 375, 10], [1920, 1080, 30]]

"""
MOT16
training : {MOT16-04, MOT16-05, MOT16-09, MOT16-10, MOT16-13}
validation :  {MOT16-02, MOT16-11}

2DMOT2015
training : 
validation : 
"""


def read_dets(is_test=False, check_occlusion=False):
    """
    Read ground truth tracking information seq-by-seq.
    Each sequence consists of float type data.
    :return: list containing data of whole sequences
    """

    global seqs

    valid_cls = [1, 2, 7]
    seq_lists = []
    for seq_name in all_seqs:

        if is_test:
            seqs.append(seq_name)
            det_path = os.path.join(seq_path, seq_name, "det", "det.txt")
        else:
            if not (seq_name in training_set or seq_name in validation_set):
                continue
            seqs.append(seq_name)
            det_path = os.path.join(seq_path, seq_name, "gt", "gt.txt")
        lines = [line.rstrip('\n').split(',') for line in open(det_path)]

        if is_test:
            seq_list = [list(list(map(float, line))) for line in lines]
        else:
            if 'MOT16' in seq_name:
                seq_list = [list(map(round, list(map(float, line)))) for line in lines if (int(line[6]) == 1) & (float(line[8]) > 0.2) & (int(line[7]) in valid_cls)]
            else:
                seq_list = [list(map(round, list(map(float, line)))) for line in lines]
        seq_lists.append(seq_list)

    seqs = np.array(seqs)
    return seq_lists


def create_fr_lists(seq_list):
    """
    Create usable lists sorted by frame number.
    format : [[fr, [id, x, y, w, h, conf, class], ...], [fr, ...], ...]
    :param seq_list: list, consists of information of each sequence
    :return: list, sorted by frame
    """
    fr_lists = [[1]]
    max_fr = 1

    seq_list = sorted(seq_list, key=operator.itemgetter(0, 1))

    for i in range(0, len(seq_list)):
        tmp_list = seq_list[i]
        cur_fr = int(tmp_list[0])
        tmp_fr_list = tmp_list[1:8]
        # Interpolate missing frames in list
        if cur_fr > max_fr:
            for fr in range(max_fr+1, cur_fr+1):
                fr_lists.append([fr])
            max_fr = cur_fr
        fr_lists[cur_fr-1].append(tmp_fr_list)

    return fr_lists


def create_id_lists(seq_list):
    """
    Create usable lists arranged by id.
    format : [[id, [fr, x, y, w, h, conf, class], ...], [id, ...], ...]
    :param seq_list: list, consists of information of each sequence
    :return: list, sorted by id
    """
    id_lists = []

    seq_list = sorted(seq_list, key=operator.itemgetter(1, 0))
    for i in range(0, len(seq_list)):
        cur_id = seq_list[i][1]
        tmp_id_list = [seq_list[i][0], *seq_list[i][2:8]]
        while len(id_lists) < cur_id:
            id_lists.append([len(id_lists)+1])

        id_lists[cur_id-1].append(tmp_id_list)

    return id_lists


def augment_bbox(bbox):
    """
    Add gaussian noise on center location & width and height
    - center noise += widht/height * N(0, 0.1)
    - width/height *= N(1, 0.1)
    :param bbox: [x, y, w, h]
    :return: augmented bounding-box
    """
    loc_aug_ratio = np.random.normal(0, 0.05)
    wh_aug_ratio = np.random.normal(1, 0.1)

    augmented_bbox = deepcopy(bbox)
    augmented_bbox[0:2] += augmented_bbox[3:4]*loc_aug_ratio
    augmented_bbox[2:4] *= wh_aug_ratio

    return augmented_bbox


def get_cropped_template(seq_name, fr, bbox):
    img = read_bgr(seq_name, fr)
    template = img[max(0, int(bbox[1])):min(img.shape[0], int(bbox[1]) + int(bbox[3])),
                 max(0, int(bbox[0])):min(img.shape[1], int(bbox[0]) + int(bbox[2]))]

    is_valid = True
    if template.shape[0] < 10 or template.shape[1] < 10:
        is_valid = False
    else:
        template = cv2.resize(template, (64, 128))

    return template, is_valid


def read_bgr(seq_name, frame_num):
    img_path = os.path.join(seq_path, seq_name, "img1", "{0:06d}.jpg".format(int(frame_num)))
    img = cv2.imread(img_path)

    return img


def normalization(img):
    norm_img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return norm_img


def create_JINet_batch(id_lists, train_val, batch_sz):
    """
    Create training batch.

    :param pos_num: positive sample number of batch
    :param neg_num: negative sample number of batch
    :param train_test: indicator, 'train' or 'test'
    :param id_lists: list, containing sequence data, sorted by id
    :return: sample_batch, anchor_batch, len_batch, label_batch
    """

    collected_num = 0

    img_batch = np.zeros((batch_sz, 128, 64, 6), dtype='float')
    label_batch = []

    # Select train&test sequence indexes
    if train_val == 'train':
        name_set = training_set
        seq_info = training_info
    else:
        name_set = validation_set
        seq_info = validation_info

    seq_idxs = [np.where(seqs == name)[0][0] for name in name_set]

    while collected_num < batch_sz:
        if collected_num % 100 == 0:
            print('collected : {}'.format(collected_num))

        name_idx = random.choice(range(len(seq_idxs)))
        seq_name = name_set[name_idx]
        seq_idx = seq_idxs[name_idx]
        id_list = id_lists[seq_idx]
        fr_rate = seq_info[name_idx][2]
        max_fr_diff = fr_rate * 3

        # Random anchor ID choice
        anchor_track_idx = random.choice(range(0, len(id_list)))
        anchor_track = id_list[anchor_track_idx][1:]
        if len(anchor_track) < 2:
            continue
        anchor_bb_idx = random.choice(range(0, len(anchor_track)))
        anchor_bb = np.array(anchor_track[anchor_bb_idx], dtype='float')

        # Random pos bb choice
        st_idx = 0
        for idx in range(anchor_bb_idx-1, -1, -1):
            if abs(anchor_track[st_idx][0] - anchor_bb[0]) > max_fr_diff:
                st_idx = idx+1
                break

        ed_idx = len(anchor_track)-1
        for idx in range(anchor_bb_idx+1, len(anchor_track), 1):
            if abs(anchor_track[ed_idx][0] - anchor_bb[0]) > max_fr_diff:
                ed_idx = idx-1
                break

        if ed_idx == st_idx:
            continue

        pos_bb_idx = random.choice(range(st_idx, ed_idx+1))
        while pos_bb_idx == anchor_bb_idx:
            pos_bb_idx = random.choice(range(st_idx, ed_idx+1))
        pos_bb = np.array(anchor_track[pos_bb_idx], dtype='float')

        # Random neg ID & bb choice
        neg_track_idx = random.choice(range(0, len(id_list)))
        while neg_track_idx == anchor_track_idx:
            neg_track_idx = random.choice(range(0, len(id_list)))
        neg_track = id_list[neg_track_idx][1:]
        if len(neg_track) == 0:
            continue
        neg_bb_idx = random.choice(range(0, len(neg_track)))
        neg_bb = np.array(neg_track[neg_bb_idx], dtype='float')

        # Get RGB templates after applying random noise

        cropped_anchor, is_valid1 = get_cropped_template(seq_name, anchor_bb[0], augment_bbox(anchor_bb[1:5]))
        cropped_pos, is_valid2 = get_cropped_template(seq_name, pos_bb[0], augment_bbox(pos_bb[1:5]))
        cropped_neg, is_valid3 = get_cropped_template(seq_name, neg_bb[0], augment_bbox(neg_bb[1:5]))

        if not (is_valid1 and is_valid2 and is_valid3):
            continue

        anchor_img = normalization(cropped_anchor)
        pos_img = normalization(cropped_pos)
        neg_img = normalization(cropped_neg)
        """
        if train_val == "train":
            cv2.imshow('pairs', np.concatenate((anchor_img, pos_img, neg_img), 1))
            cv2.waitKey(0)
        """

        img_batch[collected_num, :, :, :] = np.concatenate((anchor_img, pos_img), 2)
        collected_num += 1
        img_batch[collected_num, :, :, :] = np.concatenate((anchor_img, neg_img), 2)
        collected_num += 1

        label_batch.extend([[1, 0], [0, 1]])

    return img_batch, np.array(label_batch)


def create_LSTM_batch(max_trk_len, fr_lists, id_lists, trainval, batch_sz = 32):
    """
    Create LSTM training batch

    :param max_trk_len: maximum track length
    :param trainval: train or validation
    :return: img_batch, shp_batch, label_batch, trk_len
    """
    if trainval == 'train':
        name_set = training_set
        seq_info = training_info
    else:
        name_set = validation_set
        seq_info = validation_info

    seq_idxs = [np.where(seqs == name)[0][0] for name in name_set]

    img_batch = np.zeros((batch_sz, max_trk_len, 128, 64, 6), dtype='float')
    shp_batch = np.zeros((batch_sz, max_trk_len, 3), dtype='float')
    label_batch = []
    track_len = []

    min_len = 2
    collected_num = 0

    while collected_num < batch_sz:
        # Get an anchor sequence
        name_idx = random.choice(range(len(seq_idxs)))
        seq_idx = seq_idxs[name_idx]
        seq_name = name_set[name_idx]
        fr_rate = seq_info[name_idx][2]
        max_fr_diff = fr_rate*3

        # Get a positive anchor
        anchor_idx = random.choice([i for i in range(0, len(id_lists[seq_idx]))])
        anchor_id = id_lists[seq_idx][anchor_idx][0]
        anchor_dets = id_lists[seq_idx][anchor_idx][1:]
        anchor_det_idx = random.choice([i for i in range(min_len, len(anchor_dets))])
        anchor_det = np.array(anchor_dets[anchor_det_idx], dtype='float')

        # Make a positive track
        # Limit a searching range
        st_idx = anchor_det_idx
        for idx in range(anchor_det_idx-1, -1, -1):
            if anchor_dets[idx][0] - anchor_det[0] > max_fr_diff:
                st_idx = idx+1

        # Infeasible case
        if (anchor_det_idx - st_idx) < min_len:
            continue

        pos_pool = anchor_dets[st_idx:anchor_det_idx]
        sampling_num = random.choice([i for i in range(min_len, min(len(pos_pool), max_trk_len))])
        pos_dets = random.sample(pos_pool, sampling_num)

        # Take a negative anchor from a same frame of the positive anchor
        anchor_fr_dets = fr_lists[seq_idx][anchor_det[0]-1]
        if not len(anchor_fr_dets) > 1:
            continue
        neg_det = random.sample(anchor_fr_dets, 1)
        while neg_det[0] == anchor_id:
            neg_det = random.sample(anchor_fr_dets, 1)
        neg_det[0] = anchor_det[0]
        neg_det = np.array(neg_det, dtype='float')

        # Make batch
        anchor_img = normalization(get_cropped_template(seq_name, anchor_det[0], augment_bbox(anchor_det[1:5])))
        neg_img = normalization(get_cropped_template(seq_name, neg_det[0], augment_bbox(neg_det[1:5])))
        anchor_shp = np.array([anchor_det[0], *anchor_det[3:5]])
        neg_shp = np.array([neg_det[0], *neg_det[3:5]])
        tmp_pos_img_batch = np.zeros((0, 128, 64, 6), dtype='float')
        tmp_neg_img_batch = np.zeros((0, 128, 64, 6), dtype='float')
        tmp_pos_shp_batch = np.zeros((0, 3), dtype='float')
        tmp_neg_shp_batch = np.zeros((0, 3), dtype='float')

        cur_trk_len = len(pos_dets)
        tmp_padding_img_batch = np.zeros((max_trk_len-cur_trk_len, 128, 64, 6), dtype='float')
        tmp_padding_shp_batch = np.zeros((max_trk_len-cur_trk_len, 3), dtype='float')

        for pos_det in pos_dets:
            pos_det = np.array(pos_det, dtype='float')
            pos_img = normalization(get_cropped_template(seq_name, pos_det[0], augment_bbox(pos_det[1:5])))
            pos_shp = np.array([pos_det[0], *pos_det[3:5]], dtype='float')

            pos_shp_diff = pos_shp-anchor_shp
            pos_shp_diff[0] /= fr_rate
            pos_shp_diff[1:3] /= anchor_shp
            neg_shp_diff = neg_shp-anchor_shp
            neg_shp_diff[0] /= fr_rate
            neg_shp_diff[1:3] /= anchor_shp

            tmp_pos_img_batch = np.vstack((tmp_pos_img_batch, np.expand_dims(np.concatenate((pos_img, anchor_img), 2), 0)))
            tmp_pos_shp_batch = np.vstack((tmp_pos_shp_batch, np.expand_dims((pos_shp-anchor_shp)/anchor_shp, 0)))
            tmp_neg_img_batch = np.vstack((tmp_neg_img_batch, np.expand_dims(np.concatenate((pos_img, neg_img), 2), 0)))
            tmp_neg_shp_batch = np.vstack((tmp_neg_shp_batch, np.expand_dims((pos_shp-neg_shp)/neg_shp, 0)))

        tmp_pos_img_batch = np.vstack((tmp_padding_img_batch, tmp_pos_img_batch))
        tmp_neg_img_batch = np.vstack((tmp_padding_img_batch, tmp_neg_img_batch))
        tmp_pos_shp_batch = np.vstack((tmp_padding_shp_batch, tmp_pos_shp_batch))
        tmp_neg_shp_batch = np.vstack((tmp_padding_shp_batch, tmp_neg_shp_batch))

        img_batch[collected_num, :, :, :, :] = tmp_pos_img_batch
        shp_batch[collected_num, :, :] = tmp_pos_shp_batch
        collected_num += 1
        img_batch[collected_num, :, :, :, :] = tmp_neg_img_batch
        shp_batch[collected_num, :, :] = tmp_neg_shp_batch
        collected_num += 1

        label_batch.extend([[1, 0], [0, 1]])
        track_len.extend([cur_trk_len, cur_trk_len])

        collected_num += 2

    return img_batch, shp_batch, np.array(label_batch), track_len


class data():
    def __init__(self, is_test=False, check_occlusion=False):
        self.seq_lists = read_dets(is_test, check_occlusion=check_occlusion)
        self.fr_lists = []
        self.id_lists = []
        for i in range(0, len(self.seq_lists)):
            fr_list = create_fr_lists(self.seq_lists[i])
            self.fr_lists.append(fr_list)

            # ID based Grouping is only available from GT
            if not is_test:
                id_list = create_id_lists(self.seq_lists[i])
                self.id_lists.append(id_list)

    def get_seq_info(self, seq_name):
        name_idx = np.where(seq_name == np.array(validation_set))[0][0]

        return validation_info[name_idx]

    def get_frame_info(self, seq_name, frame_num):
        """
        format : [fr, [id, x, y, w, h], [id, x, y, w, h], ..., [id, x, y, w, h]]
        :param seq_name: name of the sequence
        :param frame_num: current frame number
        :return: bgr image, current frame bbox list
        """
        seq_idx = np.where(seqs == seq_name)[0][0]
        fr_list = self.fr_lists[seq_idx]
        cur_fr_list = fr_list[frame_num-1]
        cur_img = read_bgr(seq_name, frame_num)

        assert frame_num == cur_fr_list[0], "Frame number doesn't match!"

        return cur_img, np.array(cur_fr_list[1:])

    def get_LSTM_batch(self, max_trk_length, batch_sz, train_val):
        img_batch, shp_batch, label_batch, track_len = create_LSTM_batch(max_trk_length, self.fr_lists, self.id_lists, train_val, batch_sz)
        return img_batch, shp_batch, label_batch, track_len

    def get_JINet_batch(self, train_val, batch_sz):
        img_batch, label_batch = create_JINet_batch(self.id_lists, train_val, batch_sz)
        return img_batch, label_batch
