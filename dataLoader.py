import numpy as np
import os
import random
import cv2
import operator
from copy import deepcopy

desktop_path = os.path.expanduser("~\Desktop")
data_path = os.path.join(desktop_path, "dataset")
#seq_path = os.path.join(data_path, "2DMOT2015", "train")
#seq_path = os.path.join(data_path, "CVPR19Labels", "train")
seq_path = os.path.join(data_path, "MOT16", "train")
seqs = np.array(os.listdir(seq_path))
seq_fps_2015 = []

training_set = ['MOT16-04', 'MOT16-05', 'MOT16-09', 'MOT16-10', 'MOT16-13']
validation_set = ['MOT16-02', 'MOT16-11']

# Sequence info
# [width, height, fps]
training_info = [[1920, 1080, 30], [640, 480, 14], [1920, 1080, 30], [1920, 1080, 30], [1920, 1080, 24]]
validation_info = [[1920, 1080, 30], [1920, 1080, 30]]

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

    valid_cls = [1, 2, 7]

    seq_lists = []
    for seq_name in seqs:
        if is_test:
            det_path = os.path.join(seq_path, seq_name, "det", "det.txt")
        else:
            det_path = os.path.join(seq_path, seq_name, "gt", "gt.txt")
        lines = [line.rstrip('\n').split(',') for line in open(det_path)]

        if is_test:
            seq_list = [list(list(map(float, line))) for line in lines]
        else:
            if check_occlusion:
                seq_list = [list(map(round, list(map(float, line)))) for line in lines if (int(line[6]) == 1) & (float(line[8]) > 0.2) & (int(line[7]) in valid_cls)]
            else:
                seq_list = [list(map(round, list(map(float, line)))) for line in lines]
        seq_lists.append(seq_list)
    return seq_lists


def read_bgr(seq_name, frame_num):
    img_path = os.path.join(seq_path, seq_name, "img1", "{0:06d}.jpg".format(frame_num))
    img = cv2.imread(img_path)

    return img


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
    loc_aug_ratio = np.random.normal(0, 0.05)
    wh_aug_ratio = np.random.normal(1, 0.1)

    augmented_bbox = deepcopy(bbox)
    augmented_bbox[0:2] += augmented_bbox[3:4]*loc_aug_ratio
    augmented_bbox[2:4] *= wh_aug_ratio

    return augmented_bbox


def create_fake_det(avg_w, std_w, seq_info):
    fake_w = int(min(max(np.random.normal(avg_w, std_w), avg_w/3), seq_info[0]/10))
    wh_ratio = min(max(np.random.normal(2, 0.5), 1), 3)
    fake_h = int(fake_w * wh_ratio)
    fake_lx = random.choice(range(0, seq_info[0]-fake_w))
    fake_ty = random.choice(range(0, seq_info[1]-fake_h))

    fake_det = np.array([fake_lx, fake_ty, fake_w, fake_h, 0])

    return fake_det


def bbox_normalization(bbox, seq_info):
    aug_bbox = deepcopy(bbox)
    aug_bbox[2] += aug_bbox[0]
    aug_bbox[3] += aug_bbox[1]
    aug_bbox[0] /= seq_info[0]
    aug_bbox[2] /= seq_info[0]
    aug_bbox[1] /= seq_info[1]
    aug_bbox[3] /= seq_info[1]
    aug_bbox[4] /= seq_info[2]

    aug_bbox[0] = min(max(aug_bbox[0], 0), 1)
    aug_bbox[1] = min(max(aug_bbox[1], 0), 1)
    aug_bbox[2] = min(max(aug_bbox[2], 0), 1)
    aug_bbox[3] = min(max(aug_bbox[3], 0), 1)

    return aug_bbox


def get_cropped_template(seq_name, fr, bbox):
    img = read_bgr(seq_name, fr)
    template = img[max(0, bbox[1]):min(img.shape[0], bbox[1] + bbox[3]),
                 max(0, bbox[0]):min(img.shape[1], bbox[0] + bbox[2])]
    template = cv2.resize(template, (64, 128))

    return template


def create_triplet_batch(seq_lists, fr_lists, id_lists, train_val, batch_sz):
    """
    Create training batch.

    :param pos_num: positive sample number of batch
    :param neg_num: negative sample number of batch
    :param train_test: indicator, 'train' or 'test'
    :param seq_lists: sequence list
    :param fr_lists: list, containing sequence data, sorted by frame number
    :param id_lists: list, containing sequence data, sorted by id
    :return: sample_batch, anchor_batch, len_batch, label_batch
    """

    collected_num = 0

    anchor_batch = np.zeros((0, 128, 64, 3), dtype='uint8')
    pos_batch = np.zeros((0, 128, 64, 3), dtype='uint8')
    neg_batch = np.zeros((0, 128, 64, 3), dtype='uint8')

    # Select train&test sequence indexes
    if train_val == 'train':
        name_set = training_set
    else:
        name_set = validation_set

    seq_idxs = [np.where(seqs == name)[0][0] for name in name_set]

    while collected_num < batch_sz:
        if collected_num % 100 == 0:
            print('collected : {}'.format(collected_num))

        name_idx = random.choice(range(len(seq_idxs)))
        seq_name = name_set[name_idx]
        seq_idx = seq_idxs[name_idx]
        id_list = id_lists[seq_idx]

        # Random anchor ID choice
        anchor_track_idx = random.choice(range(0, len(id_list)))
        anchor_track = id_list[anchor_track_idx][1:]
        if len(anchor_track) < 2:
            continue
        anchor_bb_idx = random.choice(range(0, len(anchor_track)))
        anchor_bb = anchor_track[anchor_bb_idx]

        # Random pos bb choice
        pos_bb_idx = random.choice(range(0, len(anchor_track)))
        while pos_bb_idx == anchor_bb_idx:
            pos_bb_idx = random.choice(range(0, len(anchor_track)))
        pos_bb = anchor_track[pos_bb_idx]

        # Random neg ID & bb choice
        neg_track_idx = random.choice(range(0, len(id_list)))
        while neg_track_idx == anchor_track_idx:
            neg_track_idx = random.choice(range(0, len(id_list)))
        neg_track = id_list[neg_track_idx][1:]
        if len(neg_track) == 0:
            continue
        neg_bb_idx = random.choice(range(0, len(neg_track)))
        neg_bb = neg_track[neg_bb_idx]

        anchor_img = get_cropped_template(seq_name, anchor_bb[0], anchor_bb[1:5])
        pos_img = get_cropped_template(seq_name, pos_bb[0], pos_bb[1:5])
        neg_img = get_cropped_template(seq_name, neg_bb[0], neg_bb[1:5])

        #cv2.imshow('concatenated', np.hstack((anchor_img, pos_img, neg_img)))
        #cv2.waitKey(0)

        anchor_batch = np.vstack((anchor_batch, np.expand_dims(anchor_img, 0)))
        pos_batch = np.vstack((pos_batch, np.expand_dims(pos_img, 0)))
        neg_batch = np.vstack((neg_batch, np.expand_dims(neg_img, 0)))

        collected_num += 1

    return anchor_batch/255, pos_batch/255, neg_batch/255


def create_lstm_batch(max_trk_len, fr_lists, id_lists, trainval, batch_sz = 1):
    """
    Create training batch
    format : [[[x,y,w,h,fr/fps, ...,  x,y,w,h,fr/fps], ..., []],[]]
             [[(batch1)[det1, trk1_1, ..., trk1_5], ..., [det1, trk2_1, ..., trk2_5], ..., [det1, dummy_trk], [(batch2)], ...]

    :param max_trk_len: maximum track length
    :param trainval: train or validation
    :return: bb_batch, label_batch
    """
    if trainval == 'train':
        name_set = training_set
    else:
        name_set = validation_set

    seq_idxs = [np.where(seqs == name)[0][0] for name in name_set]
    seq_info = training_info

    bb_batch = []
    label_batch = []
    while len(bb_batch) < batch_sz:
        # Find anchor sequence
        name_idx = random.choice(range(len(seq_idxs)))
        seq_idx = seq_idxs[name_idx]

        # Find anchor frame
        fr_idx = random.choice([i for i in range(6, len(fr_lists[seq_idx]))])

        # Construct array of current frame det bbs
        fr_list = fr_lists[seq_idx][fr_idx]
        det_bb_pool = np.array(fr_list[1:])[:, 1:5]
        det_bb_pool = np.hstack((det_bb_pool, np.zeros((det_bb_pool.shape[0], 1))))

        # Make trk of each bbs
        det_ids = np.array(fr_list[1:])[:, 0]

        # Trk bbxs with shape [id_num, max_trk_length, bb_size]
        id_bb_pool = np.zeros((0, max_trk_len, 5))

        valid_trk_ids = []
        for id in det_ids:
            id_list = id_lists[seq_idx][id-1]
            det_arr = np.array(id_list[1:])[:, 0:5]

            # Find current frame in id_list
            anchor_idx = np.where(det_arr[:, 0] == fr_idx+1)[0][0]

            if(min(max_trk_len, anchor_idx) < 1):
                continue

            # Select random frames from id_list
            if anchor_idx > 1:
                valid_trk_ids.append(id)

                # Randomly choose artificial trk length N from [0, max_trk_len]
                bb_num = random.choice(range(0, min(max_trk_len, anchor_idx)))+1

                # Randomly choose N bbs from [anchor_idx-FPS, anchor_idx]
                trk_idxs = random.sample(range(max(0, anchor_idx - seq_info[name_idx][2]), anchor_idx), bb_num)
                trk_idxs.sort(reverse=True)
                id_bbs = np.array(id_list[1:], dtype=np.float)[trk_idxs, 1:5]

                # Attach frame difference of each trk bb from the anchor frame
                fr_dist_arr = (fr_idx + 1) - np.array(id_list[1:])[trk_idxs, 0]
                id_bbs = np.hstack((id_bbs, np.expand_dims(fr_dist_arr, 1)))

                # Random noise augmentation & normalization
                for i in range(len(id_bbs)):
                    augmented_bbox = augment_bbox(id_bbs[i])
                    normalized_bbox = bbox_normalization(augmented_bbox, seq_info[name_idx])
                    id_bbs[i] = normalized_bbox

                # Fill empty bb as zero matrices
                for i in range(0, max_trk_len - len(id_bbs)):
                    id_bbs = np.concatenate([id_bbs, np.zeros((1, 5))], axis=0)

                id_bb_pool = np.concatenate([id_bb_pool, np.expand_dims(id_bbs, 0)], axis=0)

        # Add void Trk to represent non matched pair
        id_bb_pool = np.concatenate([id_bb_pool, -1*np.ones((1, max_trk_len, 5))])

        valid_trk_ids.append(-1)
        valid_trk_ids = np.array(valid_trk_ids)

        # Make N X M matrix using det_bb_pool, id_bb_pool
        tmp_bb_batch = []
        tmp_label_batch = []

        # Randomly remove detections
        remove_pool = range(int(len(det_ids) / 10))
        if len(remove_pool) > 0:
            remove_num = random.choice(remove_pool)
            remove_idxs = random.sample(range(len(det_ids)), remove_num + 1)
            det_ids = np.delete(det_ids, remove_idxs)
            det_bb_pool = np.delete(det_bb_pool, remove_idxs, 0)

        # Randomly add False-Positives detections
        fp_num = np.random.poisson(int(len(det_ids)/2))
        avg_w = np.mean(det_bb_pool[:, 2])
        std_w = np.std(det_bb_pool[:, 2])
        for i in range(fp_num):
            fake_det = create_fake_det(avg_w, std_w, seq_info[name_idx])
            det_bb_pool = np.vstack((det_bb_pool, np.expand_dims(fake_det, axis=0)))
            det_ids = np.hstack((det_ids, 0))

        for i in range(len(det_bb_pool)):
            augmented_bbox = augment_bbox(det_bb_pool[i])
            normalized_bbox = bbox_normalization(augmented_bbox, seq_info[name_idx])
            det_bb_pool[i] = normalized_bbox

        # Shuffle the sort of each array except last row of trk (zeros)
        rand_det_idx = np.random.permutation(len(det_ids))
        rand_trk_idx = np.random.permutation(len(valid_trk_ids)-1)
        rand_trk_idx = np.append(rand_trk_idx, len(valid_trk_ids)-1)

        for det_id, det_bb in zip(det_ids[rand_det_idx], det_bb_pool[rand_det_idx]):
            for trk_id, id_bbs in zip(valid_trk_ids[rand_trk_idx], id_bb_pool[rand_trk_idx]):
                tmp_bb_batch.append(np.concatenate([np.expand_dims(det_bb, 0), id_bbs], axis=0))
                if (trk_id != -1 and trk_id == det_id):
                    tmp_label_batch.append(1)
                elif (trk_id == -1 and (det_id not in valid_trk_ids)):
                    tmp_label_batch.append(1)
                else:
                    tmp_label_batch.append(0)

        bb_batch.append(np.array(tmp_bb_batch))
        label_batch.append(np.array(tmp_label_batch))

        # Set for model input
        bb_batch = bb_batch[0].astype(float)
        bb_batch = bb_batch.reshape([1, bb_batch.shape[0], -1, ])
        label_batch = label_batch[0].astype(float)
        label_batch = label_batch.reshape([1, -1, ])

    return bb_batch, label_batch


class data():
    def __init__(self, is_test=False, check_occlusion=True):
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
        #print(seqs)
        seq_idx = np.where(seqs == seq_name)[0][0]
        #print(seq_idx)
        fr_list = self.fr_lists[seq_idx]
        cur_fr_list = fr_list[frame_num-1]
        cur_img = read_bgr(seq_name, frame_num)

        assert frame_num == cur_fr_list[0], "Frame number doesn't match!"

        return cur_img, np.array(cur_fr_list[1:])


    def get_batch(self, max_trk_length, batch_sz, train_val):
        bb_batch, label_batch = create_lstm_batch(max_trk_length, self.fr_lists, self.id_lists, train_val, batch_sz)
        return bb_batch, label_batch

    def get_triplet(self, train_val, batch_sz):
        anchor_batch, pos_batch, neg_batch = create_triplet_batch(self.seq_lists, self.fr_lists, self.id_lists, train_val, batch_sz)
        return anchor_batch, pos_batch, neg_batch

def main():
    seq_lists = read_dets()
    fr_lists = []
    id_lists = []
    for i in range(0, len(seq_lists)):
        fr_list = create_fr_lists(seq_lists[i])
        id_list = create_id_lists(seq_lists[i])
        fr_lists.append(fr_list)
        id_lists.append(id_list)
    create_training_batch(500, 500, "train", seq_lists, fr_lists, id_lists)


if __name__=="__main__":
    main()