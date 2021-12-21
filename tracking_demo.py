import os
import time
import math
import argparse

import cv2
import numpy as np
import motmetrics as mm

from utils.data_loader import Data
from config import Config
from tracking.main_tracker import Track
from dnn.neural_net import neuralNet


def track_write_result(trk_cls, _seq_name, _fr_list):
    """
    Write tracking results in MOTChallenge txt format
    """
    with open(os.path.join('results', '{}.txt'.format(_seq_name)), 'w') as file:
        for fr, trk_in_fr in enumerate(trk_cls.trk_result):
            for trk in trk_in_fr:
                tmp_write = "{}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(_fr_list[fr], trk[0], trk[1], trk[2],
                                                                        trk[3], trk[4], -1, -1, -1, -1)
                file.write(tmp_write)


def accum_evaluation(trk_cls, _gt_data, _seq_name, acc):
    """
    Accumulate evaluation data
    :param trk_cls: tracking class
    :param _gt_data: ground-truth data loader class
    :param _seq_name: sequence name
    :param acc: accumulator
    :return: None
    """
    all_name_set = np.array(_gt_data.seq_names)
    seq_idx = np.where(all_name_set == _seq_name)[0][0]
    for fr, trk_in_fr in enumerate(trk_cls.trk_result):
        _fr_list = _gt_data.fr_lists[seq_idx][fr]
        gt_id_list = [_fr_list[i][0] for i in range(1, len(_fr_list))]
        gt_bbox_list = [_fr_list[i][1:5] for i in range(1, len(_fr_list))]
        trk_id_list = [trk_in_fr[i][0] for i in range(len(trk_in_fr))]
        trk_bbox_list = [trk_in_fr[i][1:5] for i in range(len(trk_in_fr))]
        mat = mm.distances.iou_matrix(objs=np.array(gt_bbox_list), hyps=np.array(trk_bbox_list), max_iou=0.5)
        acc.update(gt_id_list, trk_id_list, mat)


def track_write_image(trk_cls, _seq_name, _data, _fr_list, trj_len=1):
    """
    Draw tracking results with trajectories
    """
    trj_list = [[] for i in range(trk_cls.max_id - 1)]
    for fr_num in range(1, len(trk_cls.trk_result)+1):
        _bgr_img, _dets = _data.get_frame_info(seq_name=_seq_name, frame_num=_fr_list[fr_num-1])
        if len(trk_cls.trk_result[fr_num-1]) > 0:
            for trk in trk_cls.trk_result[fr_num-1]:
                cv2.putText(_bgr_img, str(trk[0]), (int(trk[1]), int(trk[2])), cv2.FONT_HERSHEY_COMPLEX, 2,
                            trk[5], 2)
                cv2.rectangle(_bgr_img, (int(trk[1]), int(trk[2])), (int(trk[1] + trk[3]), int(trk[2] + trk[4])),
                              trk[5], 3)

                cur_pt = [fr_num, trk[1] + trk[3] / 2, trk[2] + trk[4]]
                for prev_pt in trj_list[trk[0] - 1][::-1]:
                    if fr_num-prev_pt[0] < trj_len:
                        cv2.line(_bgr_img, tuple(map(int, cur_pt[1:])), tuple(map(int, prev_pt[1:])), trk[5], 2)
                        cur_pt = prev_pt
                trj_list[trk[0] - 1].append([fr_num, trk[1] + trk[3] / 2, trk[2] + trk[4]])

        cv2.imwrite(os.path.join('results', _seq_name, '{}.png'.format(_fr_list[fr_num-1])), _bgr_img)


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--set_fps', action='store_true', help='Use manually configured FPS')
    parse.add_argument('--new_fps', default=5, type=int, help='Manually configured FPS (used for --set_fps)')
    parse.add_argument('--semi_on', action='store_true', help='Use semi-online mode')
    parse.add_argument('--init_mode', default='mht', type=str, help='Track initialization method (mht, delay)')
    parse.add_argument('--seqlist_name', default='validation.txt', type=str, help='Sequence list name')
    parse.add_argument('--seqlist_path', default='sequence_groups', type=str, help='Sequence group')
    parse.add_argument('--last_fr', default=-1, type=int, help='set last frame manually (-1 for default)')
    parse.add_argument('--evaluate', action='store_true', help='Evaluate result using MOT Challenge metrics')
    parse.add_argument('--draw', action='store_true', help='Draw tracking results')
    cmd_line = ['--semi_on', '--draw', '--evaluate']
    opts = parse.parse_args(cmd_line)
    print("<Argument options>")
    print(opts)

    """
    Set the semi-online mode True for better tracking performance
    1. Initialization hypothesis restoration
    2. Track interpolation
    Note : Semi-online mode outputs delayed results (Set between 5 ~ 30)
    """
    # Set the name of sequences for tracking
    seq_file_path = os.path.join(opts.seqlist_path, opts.seqlist_name)
    lines = [line.rstrip('\n').split(' ') for line in open(seq_file_path) if len(line) > 1]
    seq_names = []
    det_threshes = []
    for line in lines:
        seq_names.append(line[0])
        det_threshes.append(float(line[1]))

    print(seq_names)
    print(det_threshes)

    config = Config()
    config.semi_on = opts.semi_on
    config.init_mode = opts.init_mode
    data = Data(seq_path=config.seq_path, is_test=True, seq_names=seq_names)
    try:
        gt_data = Data(seq_path=config.seq_path, seq_names=seq_names)
    except (IOError, RuntimeError) as e:
        print(e)
        print('[Error] ground data cannot be loaded, deactivate the evaluation mode if its ON')
        gt_data = None
        opts.evaluate = False
    accs = [mm.MOTAccumulator(auto_id=True) for _ in range(len(seq_names))]

    nn_cls = neuralNet(is_test=True, train_mode='', config=config)
    # Fake inference to prevent a slow initial inference
    fake_input1 = np.ones((100, 128, 64, 6))
    fake_input2 = np.ones((100, config.lstm_len, 153))
    feature_batch = nn_cls.get_feature(fake_input1)
    likelihoods = nn_cls.get_likelihood(fake_input2)

    tot_fr = 0
    tot_time = 0
    for idx, seq_name in enumerate(seq_names):
        if not os.path.exists('results'):
            os.mkdir('results')
        if not os.path.exists(os.path.join('results', seq_name)):
            os.mkdir(os.path.join('results', seq_name))

        # Get sequence info
        seq_info = data.get_seq_info(seq_name=seq_name)

        fr_intv = 1
        # Set new frame interval based on manually configured FPS
        if opts.set_fps:
            assert seq_info[2] >= opts.new_fps, "new FPS should be equal or smaller than original FPS"
            fr_intv = math.ceil(seq_info[2]/opts.new_fps)
            seq_info[2] = math.ceil(seq_info[2]/fr_intv)

        # Get tracking parameters
        config.fps = seq_info[2]
        config.det_thresh = det_threshes[idx]
        fr_delay = config.miss_thresh-1

        _track = Track(seq_name, seq_info, data, config, fr_delay=fr_delay, nn_cls=nn_cls, visualization=False)

        print("Sequence name : {}".format(seq_name))
        print("frame interval : {}, fps : {}".format(fr_intv, seq_info[2]))
        print('thresh : (iou){:2f}, (shp){:2f}, (dist){:2f}'.format(config.assoc_iou_thresh, config.assoc_shp_thresh,
                                                                    config.assoc_dist_thresh))

        """
        Start tracking
        cur_fr : sequential frame number [1, 2, 3, ....]
        actual_fr : actual frame number of sequence [1, 1+frame_interval, 1+2*frame_interval, ...]
        if set_fps == False, cur_fr == actual_fr
        """
        fr_list = []
        cur_fr = 0
        cur_time = 0

        if opts.last_fr > 0: seq_info[-1] = opts.last_fr

        # seq_info[-1] = 80
        for actual_fr in range(1, int(seq_info[-1])+1):
            if actual_fr > 1 and (actual_fr-1) % fr_intv != 0:
                continue
            else:
                cur_fr += 1
            fr_list.append(actual_fr)

            bgr_img, dets = data.get_frame_info(seq_name=seq_name, frame_num=actual_fr)

            tmp_st_time = time.time()
            _track.track(bgr_img, dets, cur_fr)
            tmp_time = time.time() - tmp_st_time
            cur_time += tmp_time

        tot_fr += cur_fr
        tot_time += cur_time
        print('Average processing time : {} Sec/Frame'.format(cur_time/cur_fr))
        track_write_result(_track, seq_name, fr_list)
        if opts.evaluate:
            accum_evaluation(_track, gt_data, seq_name, accs[idx])
        if opts.draw:
            track_write_image(_track, seq_name, data, fr_list, trj_len=seq_info[2]*20)

    if opts.evaluate:
        mh = mm.metrics.create()
        summary = mh.compute_many(accs, metrics=mm.metrics.motchallenge_metrics, names=seq_names, generate_overall=True)
        strsummary = mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names)
        print(strsummary)
        with open(os.path.join('results', 'eval_results.txt'), 'w') as f:
            f.writelines(strsummary)

    print('Total processing time : {} Sec'.format(tot_time))
    print('Total average processing time : {} Sec/Frame'.format(tot_time / tot_fr))
