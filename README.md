# ACVLAB-SMP Challenge2025(Teamname:Devil)


## Brief Introduction & Quick Demo

This repository is the code of our **lab [ACVLAB]**(Teamname:Devil) used in SMP Challenge 2025 (http://smp-challenge.com/).

We provide all the processed features and necessary codes in this repository.

If you like to directly make the prediction of popularity scores, just clone this repository, and follow the command step by step:  
- Set up the environment which is followed by `Environment Setup` that we mention in `Detailed Instruction` chapter.
- Access **this link: https://drive.google.com/drive/u/0/folders/1Equ7FkiCf0NKg2lp4mK8mHN5OjRKVMoD**
- Download the .json files in `Caption_feature`, and put these files in `/feature_processing/caption_feature`(Remember use command `mkdir caption_feature` to create folder `caption_feature` under folder `/feature_processing` first)
- Download the .zip files in `Video_feature`, then put and unzip these files in `/feature_processing/video_feature`(Remember use command `mkdir video_feature` to create folder `video_feature` under folder `/feature_processing` first)
- Download the .zip files in `Audio_feature`, then put and unzip these files in `/feature_processing/audio_feature`(Remember use command `mkdir audio_feature` to create folder `audio_feature` under folder `/feature_processing` first)
- Excute the `python train_inference_lightgbm_5foldcrossvalidation_ensemble05.py` directly, then you can get the `test_predictions_fold_{i}.csv`(i=1,2,...,5) in `runs` folder and choose the best performance file. 

## Detailed Instruction

If you aim to reproduce the whole experiment, please run the code with the following instruction:

### 1. Environment Setup
---
Use the command as below:  
`conda create -n SMPVideo python=3.10.16`  
`conda activate SMPVideo`  
`pip install -r requirements.txt`  
You can get the virtual enviroment and packages we use


### 2. Data Preprocessing
---
We put the original dataset in the folder `/raw_data`, and you can access `/processed_data` folder to use `python clean_data.py`to get the cleaned dataset we use.  
**(For the convinence to reproduce the results, we already put the cleaned dataset in the `/processed_data` folder.)**  

Download and put the video files in `/raw_data/video_file`

### 3. Feature Engineering
---
Based on the framework we designed, we need to generate and extract the features from videos. Therefore, the codes in folder `/feature_processing` should be run at first. Before running the codes, please use 

`git clone https://github.com/DAMO-NLP-SG/VideoLLaMA3.git`  

first to download the model, and put `VideoLLaMA3-7B` at folder `/feature_processing`  

After this step, you can run `video_understanding01.ipynb`. `extract_clip_features02.py`, `audio_features03.py` in order, and get the features we need and put them in `/caption_feature`, `/video_feature`, `/audio_feature` in each folder.
**(For the convinence to reproduce the results, we already put the features we need with the google cloud link)**  


### 4. Model
---
After finishing the stpes above, you can get the features we use, so that you just excute `cluster_information_generation04.ipynb` to get the `clusters_all_types_stats_train_only_300cluster.csv`  
**(For the convinence to reproduce the results, we already put the `clusters_all_types_stats_train_only_300cluster.csv`)**  

Excute `train_inference_lightgbm_5foldcrossvalidation_ensemble05.py`, and you can get the `test_predictions_fold_{i}.csv`(i=1,2,...,5) in `runs/lightgbm_enhanced_cv_run_{current_time_str}`.  
According to MAPE performance in each fold{i}, choose the best performance `test_predictions_fold_{i}.csv` to submit.  

### Reminder

- In this project, we do video captioning with VideoLLaMA3. It is available in the open source project. If you want to reproduce this part, please follow this repository (https://github.com/DAMO-NLP-SG/VideoLLaMA3.git) and build it from source. Make sure you have install these packages for feature extraction.

- Note that the video files and feature files are too large, we didn't put it into our repository. If you want to reproduce the video captioning or video feature extraction part, please put the video files to `/raw_data/video_file`(`/raw_data/video_file/train` and `/raw_data/video_file/test`). If you want to reproduce or take a look for all the feature processing steps, please download complete file by the link we provide.

#### Environments
- PC: i9-9900K, 32GB Memory, Nvidia 3090 Ti.
- OS: Ubuntu 18.04.6 LTS (Bionic Beaver), cuda 11.5
- Software & Libs: Anaconda with python 3.10.16(**You can setup environment with 'requirements.txt'.**)

#### Copyright
- Author: Chih-Chung Hsu
e-mail: chihchung@nycu.edu.tw

- Author: Chia-Ming Lee
e-mail: zuw408421476@gmail.com

- Author: Bo-Cheng Qiu
e-mail: a36492183@gmail.com

- Author: Cheng-Jun Kang
e-mail: cjkang0601@gmail.com

- Author: I-Hsuan Wu
e-mail: wuhsuan02@gmail.com

- Author: Jun-Lin Chen
e-mail: Ryan00930207@gmail.com
