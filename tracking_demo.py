import os
import time
import data_loader as ds
from config import config
from main_tracker import track
import cv2
import math


def track_write_result(trk_cls, _seq_name, _fr_list):
    with open(os.path.join('results', _seq_name, 'results.txt'), 'w') as file:
        for fr, trk_in_fr in enumerate(trk_cls.trk_result):
            for trk in trk_in_fr:
                tmp_write = "{}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(_fr_list[fr], trk[0], trk[1], trk[2],
                                                                        trk[3], trk[4], -1, -1, -1, -1)
                file.write(tmp_write)


def track_write_image(trk_cls, _seq_name, _data, _fr_list, trj_len=1):
    trj_list = [[] for i in range(trk_cls.max_id - 1)]
    for fr_num in range(1, len(trk_cls.trk_result)+1):
        _bgr_img, _dets = data.get_frame_info(seq_name=_seq_name, frame_num=_fr_list[fr_num-1])
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


if __name__=="__main__":
    # Manually set the FPS to simulate real-time tracking
    set_fps = False
    new_fps = 5

    # Set the semi-online mode True for better tracking performance
    # 1. Initialization hypotesis restoration
    # 2. Track interpolation
    # Note : Semi-online mode will lead to a delay of a few frames (Set between 5 ~ 30)
    semi_on = True
    fr_delay = 10

    # Set the name of sequences for tracking
    seq_names = ["TUD-Stadtmitte", "PETS09-S2L1", "ETH-Bahnhof", "KITTI-13", "MOT16-02", "ETH-Crossing"]
    data = ds.data(is_test=True)

    for seq_name in seq_names:
        print("{}".format(seq_name))
        if not os.path.exists('results'):
            os.mkdir('results')
        if not os.path.exists(os.path.join('results', seq_name)):
            os.mkdir(os.path.join('results', seq_name))

        # Get sequence info
        seq_info = data.get_seq_info(seq_name=seq_name)

        assert seq_info[2] >= new_fps, "new FPS should be equal or smaller than original FPS"

        fr_intv = 1
        if set_fps:
            fr_intv = math.ceil(seq_info[2]/new_fps)
            seq_info[2] = math.ceil(seq_info[2]/fr_intv)

        print("frame interval : {}, fps : {}".format(fr_intv, seq_info[2]))

        # Get tracking parameters
        _config = config(seq_info[2])
        _config.det_thresh = 0.1
        print('thresh : (iou){:2f}, (shp){:2f}, (dist){:2f}'.format(_config.assoc_iou_thresh, _config.assoc_shp_thresh,_config.assoc_dist_thresh))

        _track = track(seq_name, seq_info, data, _config, semi_on = semi_on, fr_delay = fr_delay, visualization=False)

        # Start tracking
        # cur_fr : sequential frame number [1, 2, 3, ....]
        # actual_fr : actual frame number of sequence [1, 1+frame_interval, 1+2*frame_interval, ...]
        # if set_fps == False : cur_fr == actual_fr
        fr_list = []
        cur_fr = 0
        st_time = time.time()
        for actual_fr in range(1, seq_info[-1]+1):
            if actual_fr > 1 and (actual_fr-1) % fr_intv != 0:
                continue
            else:
                cur_fr += 1
            fr_list.append(actual_fr)
            bgr_img, dets = data.get_frame_info(seq_name=seq_name, frame_num=actual_fr)
            _track.track(bgr_img, dets, cur_fr, fr_list)

        print('Average processing time : {} Sec/Frame'.format((time.time()-st_time)/cur_fr))
        track_write_result(_track, seq_name, fr_list)
        track_write_image(_track, seq_name, data, fr_list, trj_len=100)
