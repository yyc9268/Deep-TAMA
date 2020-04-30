import os
import time
import data_loader as ds
from config import config
from main_tracker import track


def track_write(trk_result, seq_name):
    with open(os.path.join('results', seq_name, 'results.txt'), 'w') as file:
        for fr, trk_in_fr in enumerate(trk_result):
            for trk in trk_in_fr:
                tmp_write = "{}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(fr+1, trk[0], trk[1], trk[2],
                                                                        trk[3], trk[4], -1, -1, -1, -1)
                file.write(tmp_write)


if __name__=="__main__":
    # Manually set the FPS to simulate real-time tracking
    set_fps = True
    new_fps = 10

    # Set the semi-online mode True for better tracking performance
    # 1. Initialization hypotesis restoration
    # 2. Track interpolation
    # Semi-online mode will lead to a delay of a few frames (Set between 5 ~ 30)
    semi_on = False
    fr_delay = 10

    seq_names = ["PETS09-S2L1", "MOT16-02"]
    data = ds.data(is_test=True)

    for seq_name in seq_names:
        if not os.path.exists('results'):
            os.mkdir('results')
        if not os.path.exists(os.path.join('results', seq_name)):
            os.mkdir(os.path.join('results', seq_name))

        # get sequence info
        seq_info = data.get_seq_info(seq_name=seq_name)

        fr_intv = 1
        if set_fps:
            fr_intv = max(1, seq_info[2]/new_fps)
            seq_info[2] = min(seq_info[2], new_fps)

        # get tracking parameters
        _config = config(seq_info[2])
        _config.det_thresh = 0.0
        print('thresh : {}, {}'.format(_config.assoc_iou_thresh, _config.assoc_dist_thresh))

        _track = track(seq_name, seq_info, _config, semi_on = semi_on, fr_delay = fr_delay, visualization=False)
        fr_end = seq_info[-1]

        actual_fr = 0
        st_time = time.time()
        for cur_fr in range(1, seq_info[-1]+1):
            if cur_fr > 1 and (cur_fr-1) % fr_intv != 0:
                continue
            else:
                actual_fr += 1

            bgr_img, dets = data.get_frame_info(seq_name=seq_name, frame_num=cur_fr)
            _track.track(bgr_img, dets, actual_fr)

        print('{} : {} Sec'.format(seq_name, (time.time()-st_time)/actual_fr))
        track_write(_track.trk_result, seq_name)
