# Deep-TAMA

![PETS_results](https://github.com/yyc9268/Deep-TAMA/blob/master/images/PETS_results.gif)
![Stadtmitte_results](https://github.com/yyc9268/Deep-TAMA/blob/master/images/Stadtmitte_results.gif)

## Important !!!

Though the tracking is available in current version, there still remain minor errors.

We are struggling to debug and arrange the code.


## Reference

```
@inproceedings{ycyoon2018,
  title={Real-time Multiple People Tracking with Deeply Learned Candidate Selection and Person Re-identification},
  author={Young-Chul Yoon and Abhijeet Boragule and Young-min Song and Kwangjin Yoon and Moongu Jeon},
  year={2018},
  booktitle={AVSS}
}
@inproceedings{ycyoon2019,
  title={Real-time Multiple People Tracking with Deeply Learned Candidate Selection and Person Re-identification},
  author={Young-Chul Yoon Du Yong Kim and Young-min Song and Kwangjin Yoon and Moongu Jeon},
  year={2019},
  booktitle={arXiv:1907.00831}
}
```

![cvpr_award](https://github.com/yyc9268/Deep-TAMA/blob/master/images/cvpr_award.jpg)

This tracker was awarded a 3rd Prize on 4th BMTT MOTChallenge Workshop held in CVPR 2019

## Enviroment
    OS : Windows10 64bit
    CPU : Intel i5-8500 3.00GHz
    GPU : Geforce GTX Titan X
    RAM : 32 GB

## Requirements
    python 3.6
    tensorflow-gpu 2.1.0
    numpy 1.17.3
    opencv 3.4.2
    matplotlib 3.1.1
    scikit-learn 0.22.1 
    
## Tracking settings
1. Download the pre-trained models and locate in in model directory.

2. Set the public variable in 'seq_path' in data_loader.py to your own dataset path.
>> - dataset should be 'MOT/{sequence_folder-1, ..., sequence_folder-N}'.
>> - each sequence_folder should follow the MOTChallenge style (e.g., 'sequence_folder-1/{det, gt, img1}').
>> - The simplest way is just copy and paste all MOTChallenge datasets (2DMOT2015, MOT16, MOT16, MOT20, etc) in 'MOT' folder.
>> - The compatible datasets are available on [MOTChallenge](https://motchallenge.net/).

3. Perform tracking using 'tracking_demo.py'.
>> - tracking thresholds can be controlled by modifying 'config.py'.
>> - There exist two mode on-off variables in 'tracking_demo.py'.
>> - 'set_fps' can lower the FPS of the video which is for test on real-time application.
>> - 'semi_on' can increase the tracking performance though suffering a few frames delay.

## Training settings
1. Set the data as same as Tracking settings above.

2. Perform training using 'training_demo.py'.
>> - JI-Net training should be performed first.
>> - Using pre-trained JI-Net model, LSTM can be trained.

## Pre-trained models
    JI-Net : https://drive.google.com/open?id=1Xz1zEjshvPIZqi0K7WOrZOVQZ8lbQvuf
    LSTM : https://drive.google.com/open?id=1lETIh-5seYzdXEpy4WZj8Z9JPx4Fvuc3
