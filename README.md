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

After this step, you can run `video_understanding01.ipynb`. `extract_clip_features02.py`, `audio_features03.py` in order, and get the features in `/caption_feature`, `/video_feature`, `/audio_feature` in each folder.


#### 3. Model
---
After finishing the stpes above, you can get the features we use, so that you just excute `cluster_information_generation04.ipynb` and can get the .csv file

- We design the model based on **https://github.com/Daisy-zzz/CPDN** (top-1 performance at SMPD2023), and the identity-isolated split is employed.
- We consider the data-splitting strategy used in 2022/2023 (ML-based approaches) may be uncompatible to CSPN-Net (Timeseries-aware multi-modal model for social media popularity prediction), we will discuss it in our tech-report.

#### Note
- For the part of features same as last year:
We provided the two kinds of the extracted image features which are stored in *.csv format: image captioning and image 
semantic feature. Image captioning information can be extracted by executing R_04 (under tensorflow 2.0). Image semantic feature is extracted by adopting the open source project - TF_FeatureExtraction (https://github.com/tomrunia/TF_FeatureExtraction) on each image.

- In this project, we do image captioning with bilp. It is available in the open source project. If you want to reproduce this part, please follow this repository (https://github.com/salesforce/LAVIS) and build it from source. We used a pretrained blip captioning model trained with coco. We also used sentence_transformers(https://github.com/UKPLab/sentence-transformers) for getting text embedding. Make sure you have install these packages for feature extraction.

- Note that the image and feature files are too large, we didn't put it into our repository. If you want to reproduce the image captioning or image feature extraction part, please put the images to 'imgs/'('imgs/train' and 'imgs/test'). If you want to reproduce or take a look for all the feature processing steps, please download complete file by this link:  

#### Environments
- PC:  i9-9900K, 32GB Memory, Nvidia 3090 Ti.
- OS: Ubuntu 18.04.6 LTS (Bionic Beaver), cuda 11.0
- Software & Libs: Anaconda with python 3.7, Tensorflow 1.12, Tensorflow 2.0 (captioning), pytorch, sklearn, gensim, pandas, lightgbm, and pytorch-tabnet. **You can setup environment with 'requirements.txt'.**

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
