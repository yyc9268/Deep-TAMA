# Deep-TAMA

## Notice : Our paper was accepted in Elsevier Information Sciences (IF 5.910)

<img src="/images/framework.png" height="300"> 

## Reference

```
@inproceedings{ycyoon2018,
  title={Online Multi-Object Tracking with Historical Appearance Matching and Scene Adaptive Detection Filtering},
  author={Young-Chul Yoon and Abhijeet Boragule and Young-min Song and Kwangjin Yoon and Moongu Jeon},
  year={2018},
  booktitle={IEEE AVSS}
}
@inproceedings{ycyoon2020,
  title={Online Multiple Pedestrians Tracking using Deep Temporal Appearance Matching Association},
  author={Young-Chul Yoon Du Yong Kim and Young-min Song and Kwangjin Yoon and Moongu Jeon},
  year={2020},
  booktitle={Information Sciences}
}
```

<img src="/images/cvpr_award.jpg" height="400">

This tracker has been awarded a 3rd Prize on 4th BMTT MOTChallenge Workshop held in CVPR 2019

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
    
## Sample tracking dataset structure
    - Set the dataset folder as following structure
     
     MOT
      |__ TUD-Stadtmitte
      |         |__ det
      |         |__ gt
      |         |__ img1
      |
      |__ MOT16-02
      |
      |__ Custom_Seuqnce
      .
      .
      .
      
    - We recommend to copy-and-paste all MOTChallenge sequences in MOT folder
      
## Tracking settings
1. Download the pre-trained models and locate in in model directory.

2. Set the public variable in 'seq_path' in data_loader.py to your own dataset path.
    * dataset should be 'MOT/{sequence_folder-1, ..., sequence_folder-N}'.
    * each sequence_folder should follow the MOTChallenge style (e.g., 'sequence_folder-1/{det, gt, img1}').
    * The simplest way is just copy and paste all MOTChallenge datasets (2DMOT2015, MOT16, MOT16, MOT20, etc) in 'MOT' folder.
    * The compatible datasets are available on [MOTChallenge](https://motchallenge.net/).

3. Set the variable 'seqlist_name' in 'tracking_demo.py' to the proper name.
    * We have already set some sequence groups to test the tracker.
    * Add your own tracking sequence group in 'sequence_groups'.

4. Perform tracking using 'tracking_demo.py'.
    * tracking thresholds can be controlled by modifying 'config.py'.
    * There exist two mode on-off variables in 'tracking_demo.py'.
    * 'set_fps' can manipulate the FPS of the video which is for test on real-time application.
    * 'semi_on' improves the tracking performance as a trade-off of a delay of a few frames.

## Training settings
1. Set the data as same as Tracking settings above.

2. Modify the 'sequence_groups/trainval_group.json' to your own dataset
    * Note that training and validation dataset should contain 'gt' folder.

3. Perform training using 'training_demo.py'.
    * JI-Net training should be performed first.
    * Using pre-trained JI-Net model, LSTM can be trained.

## Evaluation
* The evaluation tool should be manually set by the users.
    * We recommend to use the [Matlab](https://bitbucket.org/amilan/motchallenge-devkit/src/default/) or [Python](https://github.com/cheind/py-motmetrics) evaluation tools.
    * The code produces tracking results in both txt and image format.

## Pre-trained models

<p float="left">
  <img src="/images/lstm.png" height="300"> 
</p>

* We provide pre-trained models for JI-Net and LSTM
  * Locate the downloaded models in 'model' directory
  * Download links

    JI-Net : https://drive.google.com/file/d/1Xz1zEjshvPIZqi0K7WOrZOVQZ8lbQvuf/view?usp=sharing
    
    LSTM : https://drive.google.com/file/d/1H_pcOb0HC7XAw6Xc3QLtx66a4QF1ByOn/view?usp=sharing

## Qualitative results

<p float="left">
  <img src="/images/PETS_results.gif" height="300"> 
  <img src="/images/Stadtmitte_results.gif" height="300">
</p>
