import numpy as np
import os
import random
import cv2
import operator
import json
from utils.tools import normalization, augment_bbox


class Data:
    def __init__(self, seq_path, is_test=False, seq_names=[]):
        """
        Data class initialization
        :param seq_path: path where dataset is located
        :param is_test: test mode
        :param seq_names: sequence names for test mode
        """
        if is_test:
            assert len(seq_names) > 0, "seq_names should be given during test"
        # Read sequence name and info
        self.seq_path = seq_path
        self.seq_names, self.train_idxs, self.val_idxs = self.read_seq_names(is_test, seq_lists=seq_names)
        self.seq_infos = self.read_seq_info()

        # Create lists for training
        self.seq_lists = self.read_dets(self.seq_names, is_test)
        self.fr_lists = []
        self.id_lists = []
        for i in range(0, len(self.seq_lists)):
            fr_list = self.create_fr_lists(self.seq_lists[i])
            self.fr_lists.append(fr_list)

            # ID based Grouping is only available from GT
            if not is_test:
                id_list = self.create_id_lists(self.seq_lists[i])
                self.id_lists.append(id_list)

        # Dataset collection
        self.prev_step = 0

    def read_dets(self, seq_list, is_test=False):
        """
        Read ground truth tracking information seq-by-seq.
        Each sequence consists of float type data.
        :return: list containing data of whole sequences
        """
        seq_lists = []
        for seq_name in seq_list:
            if is_test:
                det_path = os.path.join(self.seq_path, seq_name, "det", "det.txt")
            else:
                det_path = os.path.join(self.seq_path, seq_name, "gt", "gt.txt")
            lines = [line.rstrip('\n').split(',') for line in open(det_path) if len(line) > 1]

            if is_test:
                seq_list = [list(list(map(float, line))) for line in lines]
            else:
                if 'MOT16' in seq_name:
                    # human related class labels in MOT16 dataset
                    valid_cls = [1, 2, 7]
                    seq_list = [list(map(round, list(map(float, line)))) for line in lines if
                                (int(line[6]) == 1) & (float(line[8]) > 0.2) & (int(line[7]) in valid_cls)]
                else:
                    seq_list = [list(map(round, list(map(float, line)))) for line in lines]
            seq_lists.append(seq_list)

        return seq_lists

    def create_fr_lists(self, seq_list):
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
                for fr in range(max_fr + 1, cur_fr + 1):
                    fr_lists.append([fr])
                max_fr = cur_fr
            fr_lists[cur_fr - 1].append(tmp_fr_list)

        return fr_lists

    def create_id_lists(self, seq_list):
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
                id_lists.append([len(id_lists) + 1])

            id_lists[cur_id - 1].append(tmp_id_list)

        return id_lists

    def read_seq_names(self, is_test, seq_lists=[]):
        """
        Read sequence names from json file which defines training and validation set
        :param is_test: test_mode
        :param seq_lists: given seq_lists is returned for test mode
        :return: sequence names, training data indexes, validation data indexes
        """
        seq_names = []
        train_idxs = []
        val_idxs = []
        if is_test or len(seq_lists) > 0:
            seq_names = seq_lists
        else:
            with open(os.path.join('sequence_groups', 'trainval_group.json')) as json_file:
                json_data = json.load(json_file)

                seq_names.extend(json_data['train'])
                seq_names.extend(json_data['validation'])
                train_idxs = [i for i in range(len(json_data['train']))]
                val_idxs = [len(json_data['train']) + i for i in range(len(json_data['validation']))]

        return seq_names, train_idxs, val_idxs

    def read_seq_info(self):
        """
        Read sequence info
        :return: [width, height, fps, total_frame_num]
        """
        seq_infos = []
        for seq_name in self.seq_names:
            # Since MOT17 shares same sequences with MOT16
            if "MOT17" in seq_name:
                seq_name = "MOT16-"+seq_name.split('-')[1]
            # sequence info format [width, height, fps, total_frame_num]
            with open(os.path.join('sequence_infos', '{}.txt'.format(seq_name))) as seq_info_file:
                line = seq_info_file.readline()
                seq_infos.append(list(map(float, line.split(','))))

        return seq_infos

    def get_seq_info(self, seq_name):
        name_idx = np.where(seq_name == np.array(self.seq_names))[0][0]

        return self.seq_infos[name_idx]

    def get_frame_info(self, seq_name, frame_num):
        """
        format : [fr, [id, x, y, w, h], [id, x, y, w, h], ..., [id, x, y, w, h]]
        :param seq_name: name of the sequence
        :param frame_num: current frame number
        :return: bgr image, current frame bbox list
        """
        seq_idx = np.where(np.array(self.seq_names) == seq_name)[0][0]
        fr_list = self.fr_lists[seq_idx]

        cur_img = self.read_bgr(seq_name, frame_num)

        if frame_num > len(fr_list):
            return cur_img, np.array([])
        else:
            cur_fr_list = fr_list[frame_num-1]
            assert frame_num == cur_fr_list[0], "Frame number doesn't match!"
            return cur_img, np.array(cur_fr_list[1:])

    def get_cropped_template(self, seq_name, fr, bbox):
        img = self.read_bgr(seq_name, fr)
        template = img[max(0, int(bbox[1])):min(img.shape[0], int(bbox[1]) + int(bbox[3])),
                   max(0, int(bbox[0])):min(img.shape[1], int(bbox[0]) + int(bbox[2]))]

        is_valid = True
        if template.shape[0] < 10 or template.shape[1] < 10:
            is_valid = False
        else:
            template = cv2.resize(template, (64, 128))

        return template, is_valid

    def read_bgr(self, seq_name, frame_num):
        img_path = os.path.join(self.seq_path, seq_name, "img1", "{0:06d}.jpg".format(int(frame_num)))
        img = cv2.imread(img_path)

        return img

    def print_progress_bar(self, cur_sz, max_sz, step_sz=10):
        step_intv = max_sz // step_sz
        if cur_sz // step_intv > self.prev_step:
            self.prev_step += 1
            print('({:06d} / {:06d}) ||'.format(cur_sz, max_sz), end='')
            for _ in range(self.prev_step):
                print('==', end='')
            for _ in range(self.prev_step, step_sz):
                print('  ', end='')
            print('||')

    def create_jinet_batch(self, batch_sz, train_val):
        """
        Create JI-Net training batch.
        """

        collected_num = 0

        img_batch = np.zeros((batch_sz, 128, 64, 6), dtype='float')
        label_batch = []

        # Select train&test sequence indexes
        all_name_set = np.array(self.seq_names)
        if train_val == "train":
            name_set = np.array(self.seq_names)[self.train_idxs]
            seq_info = np.array(self.seq_infos)[self.train_idxs]
        else:
            name_set = np.array(self.seq_names)[self.val_idxs]
            seq_info = np.array(self.seq_infos)[self.val_idxs]

        seq_idxs = [np.where(all_name_set == name)[0][0] for name in name_set]
        self.prev_step = 0
        while collected_num < batch_sz:
            self.print_progress_bar(collected_num, batch_sz, step_sz=20)

            name_idx = random.choice(range(len(seq_idxs)))
            seq_name = name_set[name_idx]
            seq_idx = seq_idxs[name_idx]
            id_list = self.id_lists[seq_idx]
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
            for idx in range(anchor_bb_idx - 1, -1, -1):
                if abs(anchor_track[st_idx][0] - anchor_bb[0]) > max_fr_diff:
                    st_idx = idx + 1
                    break

            ed_idx = len(anchor_track) - 1
            for idx in range(anchor_bb_idx + 1, len(anchor_track), 1):
                if abs(anchor_track[ed_idx][0] - anchor_bb[0]) > max_fr_diff:
                    ed_idx = idx - 1
                    break

            if ed_idx == st_idx:
                continue

            pos_bb_idx = random.choice(range(st_idx, ed_idx + 1))
            while pos_bb_idx == anchor_bb_idx:
                pos_bb_idx = random.choice(range(st_idx, ed_idx + 1))
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
            cropped_anchor, is_valid1 = self.get_cropped_template(seq_name, anchor_bb[0], augment_bbox(anchor_bb[1:5]))
            cropped_pos, is_valid2 = self.get_cropped_template(seq_name, pos_bb[0], augment_bbox(pos_bb[1:5]))
            cropped_neg, is_valid3 = self.get_cropped_template(seq_name, neg_bb[0], augment_bbox(neg_bb[1:5], very_noisy = True))

            if not (is_valid1 and is_valid2 and is_valid3):
                continue

            anchor_img = normalization(cropped_anchor)
            pos_img = normalization(cropped_pos)
            neg_img = normalization(cropped_neg)

            img_batch[collected_num, :, :, :] = np.concatenate((anchor_img, pos_img), 2)
            collected_num += 1
            img_batch[collected_num, :, :, :] = np.concatenate((anchor_img, neg_img), 2)
            collected_num += 1

            label_batch.extend([[1, 0], [0, 1]])

        return img_batch, np.array(label_batch)

    def create_deeptama_batch(self, max_trk_len, batch_sz, train_val):
        """
        Create LSTM training batch
        """

        all_name_set = np.array(self.seq_names)
        if train_val == "train":
            name_set = np.array(self.seq_names)[self.train_idxs]
            seq_info = np.array(self.seq_infos)[self.train_idxs]
        else:
            name_set = np.array(self.seq_names)[self.val_idxs]
            seq_info = np.array(self.seq_infos)[self.val_idxs]

        seq_idxs = [np.where(all_name_set == name)[0][0] for name in name_set]

        img_batch = np.zeros((batch_sz, max_trk_len, 128, 64, 6), dtype='float')
        shp_batch = np.zeros((batch_sz, max_trk_len, 3), dtype='float')
        label_batch = []
        track_len = []

        min_len = 1
        collected_num = 0
        self.prev_step = 0
        while collected_num < batch_sz:
            self.print_progress_bar(collected_num, batch_sz, step_sz=20)

            # Get an anchor sequence
            name_idx = random.choice(range(len(seq_idxs)))
            seq_idx = seq_idxs[name_idx]
            seq_name = name_set[name_idx]
            fr_rate = seq_info[name_idx][2]
            max_fr_diff = fr_rate * 2

            # Get a positive anchor
            anchor_idx = random.choice([i for i in range(0, len(self.id_lists[seq_idx]))])
            recur_cnt = 0
            while len(self.id_lists[seq_idx][anchor_idx][1:]) <= min_len and recur_cnt <= 5:
                anchor_idx = random.choice([i for i in range(0, len(self.id_lists[seq_idx]))])
                recur_cnt += 1
            if recur_cnt > 5:
                # print('[Warning] failed to find positive anchor idx')
                continue

            anchor_id = self.id_lists[seq_idx][anchor_idx][0]
            anchor_dets = self.id_lists[seq_idx][anchor_idx][1:]

            anchor_det_idx = random.choice([i for i in range(min_len, len(anchor_dets))])
            anchor_det = np.array(anchor_dets[anchor_det_idx], dtype='float')

            # Make a positive track
            # Limit a searching range
            st_idx = 0
            for idx in range(anchor_det_idx - 1, -1, -1):
                if anchor_dets[idx][0] - anchor_det[0] > max_fr_diff:
                    st_idx = idx + 1

            # Infeasible case
            if (anchor_det_idx - st_idx) <= min_len:
                continue

            pos_pool = anchor_dets[st_idx:anchor_det_idx]
            sampling_num = random.choice([i for i in range(min_len, min(len(pos_pool), max_trk_len)+1)])
            pos_dets = random.sample(pos_pool, sampling_num)
            pos_dets.sort(key=lambda x: x[0])

            # Take a negative anchor from a same frame of the positive anchor
            anchor_fr_dets = self.fr_lists[seq_idx][int(anchor_det[0]) - 1][1:]
            if not len(anchor_fr_dets) > 1:
                continue
            neg_det = random.sample(anchor_fr_dets, 1)[0]

            recur_cnt = 1
            while neg_det[0] == anchor_id and recur_cnt <= 5:
                neg_det = random.sample(anchor_fr_dets, 1)[0]
                recur_cnt += 1
            if recur_cnt > 5:
                # print('[Warning] failed to find negative anchor idx')
                continue
            neg_det[0] = anchor_det[0]
            neg_det = np.array(neg_det, dtype='float')

            # Make batch
            anchor_det[1:5] = augment_bbox(anchor_det[1:5])
            anchor_img, is_valid1 = self.get_cropped_template(seq_name, anchor_det[0], anchor_det[1:5])
            neg_det[1:5] = augment_bbox(neg_det[1:5], very_noisy=True)
            neg_img, is_valid2 = self.get_cropped_template(seq_name, neg_det[0], neg_det[1:5])

            if not (is_valid1 and is_valid2):
                continue

            anchor_img = normalization(anchor_img)
            neg_img = normalization(neg_img)

            anchor_shp = np.array([anchor_det[0], *anchor_det[3:5]])
            neg_shp = np.array([neg_det[0], *neg_det[3:5]])
            tmp_pos_img_batch = np.zeros((0, 128, 64, 6), dtype='float')
            tmp_neg_img_batch = np.zeros((0, 128, 64, 6), dtype='float')
            tmp_pos_shp_batch = np.zeros((0, 3), dtype='float')
            tmp_neg_shp_batch = np.zeros((0, 3), dtype='float')

            cur_trk_len = len(pos_dets)
            tmp_padding_img_batch = np.zeros((max_trk_len - cur_trk_len, 128, 64, 6), dtype='float')
            tmp_padding_shp_batch = np.zeros((max_trk_len - cur_trk_len, 3), dtype='float')

            is_valid3 = True

            for idx, pos_det in enumerate(pos_dets):
                pos_det = np.array(pos_det, dtype='float')
                pos_det[1:5] = augment_bbox(pos_det[1:5])
                if train_val == 'train' and len(pos_dets) > 3 and idx == len(pos_dets)-1:
                    pos_det[1:5] = augment_bbox(pos_det[1:5], very_noisy=True)
                else:
                    pos_det[1:5] = augment_bbox(pos_det[1:5])

                pos_img, is_valid = self.get_cropped_template(seq_name, pos_det[0], pos_det[1:5])
                if not is_valid:
                    is_valid3 = False
                    break

                pos_img = normalization(pos_img)
                pos_shp = np.array([pos_det[0], *pos_det[3:5]], dtype='float')

                pos_shp_diff = pos_shp - anchor_shp
                pos_shp_diff[0] /= fr_rate
                pos_shp_diff[1:3] /= anchor_shp[1:3]
                neg_shp_diff = pos_shp - neg_shp
                neg_shp_diff[0] /= fr_rate
                neg_shp_diff[1:3] /= neg_shp[1:3]

                tmp_pos_img_batch = np.vstack(
                    (tmp_pos_img_batch, np.expand_dims(np.concatenate((pos_img, anchor_img), 2), 0)))
                tmp_pos_shp_batch = np.vstack(
                    (tmp_pos_shp_batch, np.expand_dims((pos_shp - anchor_shp) / anchor_shp, 0)))
                tmp_neg_img_batch = np.vstack(
                    (tmp_neg_img_batch, np.expand_dims(np.concatenate((pos_img, neg_img), 2), 0)))
                tmp_neg_shp_batch = np.vstack((tmp_neg_shp_batch, np.expand_dims((pos_shp - neg_shp) / neg_shp, 0)))

            if not is_valid3:
                continue

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

        return img_batch, shp_batch, np.array(label_batch), track_len

    def get_deeptama_batch(self, max_trk_length, batch_sz, train_val):
        """
        Get input batch for LSTM
        :param max_trk_length: maximum length of each track sequence
        :param batch_sz: batch size
        :param train_val: training or validation mode
        :return: image batch, shape batch, label batch, track length (list)
        """
        img_batch, shp_batch, label_batch, track_len = self.create_deeptama_batch(max_trk_length, batch_sz, train_val)
        return img_batch, shp_batch, label_batch, track_len

    def get_jinet_batch(self, batch_sz, train_val):
        """
        Get JI-Net input batch
        :param batch_sz: batch size
        :param train_val: training or validation mode
        :return: image batch, label batch
        """
        img_batch, label_batch = self.create_jinet_batch(batch_sz, train_val)
        return img_batch, label_batch
