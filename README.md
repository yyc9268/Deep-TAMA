# Deep-TAMA 2021

#### The paper was accepted in Elsevier Information Sciences (IF 6.795)
- arXiv paper link : [https://arxiv.org/abs/1907.00831](https://arxiv.org/abs/1907.00831)

<img src="/images/framework.png" height="300"> 

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
    * Tracking thresholds can be controlled by modifying ```config.py```.
    * List of significant commandline arguments in ```tracking_demo.py```.
        * ```set_fps``` : manipulates an FPS and drop frames of videos
        * ```semi_on``` : improves a tracking performance using interpolation and restoration
        * ```init_mode``` : ```mht``` (faster, geometry-based), ```delayed``` (geometry + appearance)
        * ```gating_mode``` : ```iou``` (intersection-over-union), ```maha``` (pose + shape)
        * ```only_motion``` : Do not use appearance matching

## Training settings
1. Set the data as same as 'Tracking settings' above.

2. Modify the ```sequence_groups/trainval_group.json``` to your own dataset
    * Training and validation dataset should have ```gt``` folder.

3. Perform training using ```training_demo.py```.
    * JI-Net training should be done first.
    * Using JI-Net model as a feature extractor, LSTM can be trained.

## Evaluation
* Evaluated performance on sequence group ```validation.txt```.
    * The highlighted performance can be achieved by default demo setting.

|Setting|MOTA|IDF1|MT|PT|ML|IDs|FM|
|---|---|---|---|---|---|---|---|
|Baseline (no appearance feature)|38.9%|45.0%|40|138|166|128|648|
|Baseline + semi-online|40.8%|47.4%|76|123|145|126|421|
|**Deep-TAMA + semi-online**|43.0%|49.4%|80|114|150|122|364|

* For evaluation, set the command line argument ```--evaluate``` in ```tracking_demo.py```
* Currently, our code doesn't support HOTA. For HOTA, check [TrackEval](https://github.com/JonathonLuiten/TrackEval).
* The code produces tracking results in both txt and image format.


## Pre-trained models
* Validation accuracy and loss comparison
    * Both of JI-Net and LSTM used cross-entropy loss for training
    <p float="left">
      <img src="/images/acc_comparison.png" height="200">
      <img src="/images/loss_comparison.png" height="200">
    </p>
* We provide pre-trained models for JI-Net and LSTM
  * Locate the downloaded models in ```model``` directory
  * Download links
    * JI-Net : https://drive.google.com/file/d/1Lreh8FxDYYx3ymgm9B4TuD_lz0y7Vi7C/view?usp=sharing
    * LSTM : https://drive.google.com/file/d/1KbUHnfVSRkuV6SMFCdvs7U5PuRoR2Dxi/view?usp=sharing

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
@inproceedings{ycyoon2021,
  title={Online Multiple Pedestrians Tracking using Deep Temporal Appearance Matching Association},
  author={Young-Chul Yoon Du Yong Kim and Young-min Song and Kwangjin Yoon and Moongu Jeon},
  year={2021},
  booktitle={Information Sciences}
}
```

<img src="/images/cvpr_award.jpg" height="400">

* This tracker has been awarded a 3rd Prize on [4th BMTT MOTChallenge Workshop](https://motchallenge.net/results/CVPR_2019_Tracking_Challenge/) held in CVPR 2019
* If you have any question, feel free to contact by [yyc9268@gmail.com](yyc9268@gmail.com)