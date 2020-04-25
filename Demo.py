import os
import time
import dataLoader as ds
from Config import Config
from tracker import Track

if __name__=="__main__":
    seq_names = ["MOT17-02-SDP"]
    data = ds.data(is_test=True)

    for seq_name in seq_names:
        if not os.path.exists('results'):
            os.mkdir('results')
        if not os.path.exists(os.path.join('results', seq_name)):
            os.mkdir(os.path.join('results', seq_name))

        # get sequence info
        seq_info = data.get_seq_info(seq_name=seq_name)

        # get tracking parameters
        config = Config(seq_info[-1])

        track = Track(seq_name, seq_info, config, visualization=False)
        fr_end = seq_info[-1]
        for cur_fr in range(1, seq_info[-1]+1):
            st_time = time.time()
            _bgr_img, _dets = data.get_frame_info(seq_name=seq_name, frame_num=cur_fr)
            track.track(_bgr_img, _dets, cur_fr)
            print('{} Sec'.format(time.time()-st_time))