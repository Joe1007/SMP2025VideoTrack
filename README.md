# NCKU_ACVLAB-SMPChallenge2025


## Brief Introduction & Quick Demo

This repository is the code of our **team [NCKU_ACVLAB]** used in SMP Challenge 2025 (http://smp-challenge.com/).

We provide all the processed features and necessary codes in this repository.

If you like to directly make the prediction of popularity scores, just clone this repository, and follow the command step by step:  
- Access **this link: https://drive.google.com/drive/u/0/folders/1Equ7FkiCf0NKg2lp4mK8mHN5OjRKVMoD**
- Download the .json files in `Caption_feature`, and put these files in `/feature_processing/caption_feature`
- Download the .zip files in `Video_feature`, then put and unzip these files in `/feature_processing/video_feature`
- Download the .zip files in `Audio_feature`, then put and unzip these files in `/feature_processing/audio_feature`
- Excute the `train_inference_lightgbm_5foldcrossvalidation_ensemble05.py` directly, and you can get the `submission.csv` with the best performance we choose

## Detailed Instruction

If you aim to reproduce the whole experiment, please run the code with the following instruction:

### 1. Data Preprocessing
---
We put the original dataset in the folder `/raw_data`, and you just use `xxx.py` to get the cleaned dataset at `/processed_data`.  
**For the convinence to reproduce the results, we already put the cleaned dataset in the  `/processed_data` folder.**

### 2. Feature Engineering
---
Based on the framework we designed, we need to generate and extract the features from videos. Therefore, the codes in folder `/feature_processing` should be run at first. Before running the codes, please use 

`git clone https://github.com/DAMO-NLP-SG/VideoLLaMA3.git`  

first to download the model, and put `VideoLLaMA3-7B` at folder `/feature_processing`  

After this step, you can run `video_understanding01.ipynb`. `extract_clip_features02.py`, `audio_features03.py` in order, and get the features we need in `/caption_feature`, `/video_feature`, `/audio_feature` in each folder.


### 3. Model
---
After finishing the stpes above, you can get the features we use, so that you just excute `cluster_information_generation04.ipynb` to get the `.csv` file  
Excute `train_inference_lightgbm_5foldcrossvalidation_ensemble05.py`, and you can get the `test_prediction.csv` in `runs/lightgbm_enhanced_cv_run_{current_time_str}`. According to MAPE performance in each fold, choose the best performance `test_prediction.csv` to submit.  

### Reminder
- For the part of features same as last year:
We provided the two kinds of the extracted image features which are stored in *.csv format: image captioning and image 
semantic feature. Image captioning information can be extracted by executing R_04 (under tensorflow 2.0). Image semantic feature is extracted by adopting the open source project - TF_FeatureExtraction (https://github.com/tomrunia/TF_FeatureExtraction) on each image.

- In this project, we do video captioning with VideoLLaMA3. It is available in the open source project. If you want to reproduce this part, please follow this repository (https://github.com/DAMO-NLP-SG/VideoLLaMA3.git) and build it from source. Make sure you have install these packages for feature extraction.

- Note that the video files and feature files are too large, we didn't put it into our repository. If you want to reproduce the video captioning or video feature extraction part, please put the video files to '/raw_data/video_file'(`/raw_data/video_file/train` and `/raw_data/video_file/test`). If you want to reproduce or take a look for all the feature processing steps, please download complete file by the link we provide.  

#### Environments
- PC:  i9-9900K, 32GB Memory, Nvidia 3090 Ti.
- OS: Ubuntu 18.04.6 LTS (Bionic Beaver), cuda 11.5
- Software & Libs: Anaconda with python 3.10.16, **You can setup environment with 'requirements.txt'.**

#### Copyright
- Author: Chih-Chung Hsu
e-mail: cchsu@gs.ncku.edu.tw

- Author: Chia-Ming Lee
e-mail: zuw408421476@gmail.com

- Author: Yu-Fan Lin
e-mail: aas12as12as12tw@gmail.com

- Author: Yi-Shiuan Chou
e-mail: nelly910421@gmail.com

- Author: Chih-Yu Jian
e-mail: ru0354m3@gmail.com

- Author: Chi-Han Tsai
e-mail: fateplsf567@gmail.com
