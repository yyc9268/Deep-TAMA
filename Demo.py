import os
import time
import dataLoader as ds
from Config import Config
from tracker import Track

if __name__=="__main__":

    # Manually set the FPS to simulate real-time tracking
    set_fps = True
    new_fps = 10

    seq_names = ["MOT16-02"]
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
            fr_intv = int(seq_info[2]/new_fps)
            seq_info[2] = new_fps

        # get tracking parameters
        config = Config(seq_info[2])

        track = Track(seq_name, seq_info, config, visualization=False)
        fr_end = seq_info[-1]

        actual_fr = 0
        for cur_fr in range(1, seq_info[-1]+1):
            if cur_fr > 1 and (cur_fr-1) % fr_intv != 0:
                continue
            else:
                actual_fr += 1

            st_time = time.time()
            bgr_img, dets = data.get_frame_info(seq_name=seq_name, frame_num=cur_fr)
            track.track(bgr_img, dets, actual_fr)
            print('{} Sec'.format(time.time()-st_time))