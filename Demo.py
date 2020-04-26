import os
import time
import dataLoader as ds
from Config import Config
from tracker import Track

if __name__=="__main__":
    seq_names = ["PETS09-S2L1"]
    data = ds.data(is_test=True)

    for seq_name in seq_names:
        if not os.path.exists('results'):
            os.mkdir('results')
        if not os.path.exists(os.path.join('results', seq_name)):
            os.mkdir(os.path.join('results', seq_name))

        # get sequence info
        seq_info = data.get_seq_info(seq_name=seq_name)
        print(seq_info)
        # get tracking parameters
        config = Config(seq_info[-1])

        track = Track(seq_name, seq_info, config, visualization=False)
        fr_end = seq_info[-1]
        for cur_fr in range(1, seq_info[-1]+1):
            st_time = time.time()
            bgr_img, dets = data.get_frame_info(seq_name=seq_name, frame_num=cur_fr)
            track.track(bgr_img, dets, cur_fr)
            print('{} Sec'.format(time.time()-st_time))