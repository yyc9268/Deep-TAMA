# Deep-TAMA 2021

#### The paper was accepted in Elsevier Information Sciences (IF 6.795)
- arXiv paper link : [https://arxiv.org/abs/1907.00831](https://arxiv.org/abs/1907.00831)

<img src="/images/framework.png" height="300"> 

## To Do
* ~~Code refactoring (12/12)~~
* ~~Two initialization methods (MHT, association-based)~~
* ~~Tracking evaluation code~~
* Demo tracking sequence update

## Enviroment
    OS : Windows10 64bit (Verified to works fine on Ubuntu 18.04)
    CPU : Intel i5-8500 3.00GHz
    GPU : Geforce GTX Titan X (Works on GPU with smaller memory size >= 5GB)
    RAM : 32 GB

## Requirements
    python 3.6
    tensorflow-gpu 2.1.0 (strict!)
    numpy 1.17.3
    opencv 3.4.2
    matplotlib 3.1.1
    scikit-learn 0.22.1
    motmetrics
    
## Sample tracking dataset structure
    - Set the dataset folder as following structure
      ex) MOT
          |__ TUD-Stadtmitte
          |         |__ det
          |         |__ gt
          |         |__ img1
          |__ MOT16-02
          |__ Custom_Seuqnce
          .
          .
    - We recommend to copy-and-paste all MOTChallenge sequences in MOT folder
      
## Tracking settings
1. Download the pre-trained models and locate in in ```model``` directory.

2. Set the variable ```seq_path``` in ```config.py``` to your own dataset path.
    * Dataset should be 'MOT/{sequence_folder-1, ..., sequence_folder-N}'.
    * Each of sequence_folder should follow the MOTChallenge style (e.g., 'sequence_folder-1/{det, gt, img1}').
    * The simplest way is just copy and paste all MOTChallenge datasets (2DMOT2015, MOT16, MOT16, MOT20, etc) in ```MOT``` folder.
    * The compatible datasets are available on [MOTChallenge](https://motchallenge.net/).

3. Set the variable ```seqlist_name``` in ```racking_demo.py``` to the proper name.
    * Exemplar sequence groups are provided.
    * Add your own tracking sequence group in ```sequence_groups```.

4. Perform tracking using ```tracking_demo.py```.
    * tracking thresholds can be controlled by modifying ```config.py```.
    * There exist a few mode on-off variables in ```tracking_demo.py```.
        * ```set_fps``` : manipulates an FPS and drop frames of videos
        * ```semi_on``` : improves a tracking performance using interpolation and restoration
        * ```init_mode``` : ```mht``` (faster, geometry-based), ```delayed``` (geometry + appearance)

## Training settings
1. Set the data as same as 'Tracking settings' above.

2. Modify the ```sequence_groups/trainval_group.json``` to your own dataset
    * Training and validation dataset should have ```gt``` folder.

3. Perform training using ```training_demo.py```.
    * JI-Net training should be performed first.
    * Using pre-trained JI-Net model, LSTM can be trained.

## Evaluation
* For evaluation, set the command line argument ```--evaluate``` in ```tracking_demo.py```
* Currently, our code doesn't support HOTA.
    * For HOTA evaluation, check [TrackEval](https://github.com/JonathonLuiten/TrackEval).
* The code produces tracking results in both txt and image format.

## Pre-trained models
* LSTM validation loss and accuracy
    <p float="left">
      <img src="/images/lstm.png" height="250"> 
    </p>

* We provide pre-trained models for JI-Net and LSTM
  * Locate the downloaded models in ```model``` directory
  * Download links

    JI-Net : https://drive.google.com/file/d/1VnJoyUOuDPbP82kgqznoZlKSaZ7QdaiZ/view?usp=sharing
    
    LSTM : https://drive.google.com/file/d/1jkGdbSqfP7Pc9CyFxNT6aAAam1pWyA_X/view?usp=sharing

## Qualitative results

<p float="left">
  <img src="/images/PETS_results.gif" height="300"> 
  <img src="/images/Stadtmitte_results.gif" height="300">
</p>

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
  year={2021},
  booktitle={Information Sciences}
}
```

<img src="/images/cvpr_award.jpg" height="400">

This tracker has been awarded a 3rd Prize on [4th BMTT MOTChallenge Workshop](https://motchallenge.net/results/CVPR_2019_Tracking_Challenge/) held in CVPR 2019

