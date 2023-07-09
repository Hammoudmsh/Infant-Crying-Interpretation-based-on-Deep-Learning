# Application of deep learning methods for the investigation of the neurological disease diagnosed through NIR eye video

## Table of contents
* [Introduction](#introduction)
* [Technologies](#technologies)
* [requirements](#requirements)
* [Launch](#launch)
* [Results](#results)

## Introduction
Deep learning is extensively used in neurological applications as decision support for doctors. Parkin disease(PD) and Progressive Supranuclear Palsy(PSP) are crucial neurodegenerative diseases that have similar symptoms. The goal of this work is to classify PD, PSP, and Healthy Control(HC) with high accuracy through eye movements. The paper is <a href="" target="_blank">Paper(will be added)</a>.
**Research questions:**
- 
- 
-

## Technologies
* Computer vision
* Deep learning
* time-series analysis

## Requirements
1- create a virtual environment with conda and activate it
```
conda create --name sk_pro python=3.6
conda activate sk_pro
pip install -r requirements_paper.txt
```
## Launch
### download data

* Manual downloading:
<a href="https://github.com/gveres/donateacry-corpus" target="_blank">donateacry-corpus</a>
* Automatic run scripts, to download opened, navgaze, natural, and NN human mouse datasets.
```
python download_data.py
```
### Data augmentation 
```
python data_aug.py
```
### Transform audio into images and generate Mel-spectrogram
```
python audio_mfcc_tsi
python generateSpectrograms.py
```

### train
```
python train.py --dataset data/aug_GADF_280_dataset/ --epochs 500 --lr 0.0001 --file2read 0 --es 1 --batch_size 16 --output output_id --experiment_tag output_tag
```
2 - Commands
* train
```
python train.py --DS_NAME 'NN_human_mouse_eyes' --epochs 1 --batch_size 128 --lr 1e-05 --es 1 --wanted 'p' --file2read 100 --backbon 'mine' --weights '[1,0,0,0]' --output output_id
```

# Results
## 
![image](https://github.com/Hammoudmsh/Neurological-disease-diagonisis-through-eye-movements/assets/57059181/da381039-1069-4e43-b537-d98d35c5a0b9)
<br />

