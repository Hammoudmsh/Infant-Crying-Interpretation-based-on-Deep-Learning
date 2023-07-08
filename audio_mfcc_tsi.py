#!/usr/bin/env python
# coding: utf-8

# In[129]:


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot
from pyts.approximation import PiecewiseAggregateApproximation
import pandas as pd
import glob
from tqdm.auto import tqdm
import pathlib
import cv2
import numpy as np
import tensorflow as tf
import librosa
import numpy as np
import os
def normalize(data):
    xmax, xmin =  data.max(), data.min()
    zi = 2 * ((data - xmin) / (xmax - xmin)) - 1
    return zi
    

def audio_mfcc(audio, n_mfcc = 50):
    signal, sr = librosa.load(audio, res_type = "kaiser_fast")
    mfcc_signal = np.mean(librosa.feature.mfcc(y = signal, sr = sr, n_mfcc  =n_mfcc), axis = 0)
    return mfcc_signal

def approximate_ts(X, window_size):
    paa = PiecewiseAggregateApproximation(window_size=window_size)
    X_paa = paa.transform(X)
    return X_paa

def timeSeriesToImage(ts, size_x = None, kind = "GADF", window_size = 0):
    if window_size != 0:
        ts = approximate_ts(ts.reshape(1, -1) , window_size)
        ts = ts.reshape(-1,1)
    gasf = GramianAngularField(method='summation')
    gadf = GramianAngularField(method='difference')
    mtf = MarkovTransitionField()
    rp = RecurrencePlot()

    rp = RecurrencePlot()

    if kind == "GADF":
        img = gadf.fit_transform(pd.DataFrame(ts).T)[0]
    elif kind == "GASF":
        img = gasf.fit_transform(pd.DataFrame(ts).T)[0]
    elif kind == "MTF":
        img = mtf.fit_transform(pd.DataFrame(ts).T)[0]
#         img = transformer.transform(ts)
    elif kind == "RP":
        img = rp.fit_transform(pd.DataFrame(ts).T)[0]
#         img = transformer.transform(ts)
    elif kind == "RGB_GAF":
        gasf_img = gasf.transform(pd.DataFrame(ts).T)[0]
        gadf_img = gadf.transform(pd.DataFrame(ts).T)[0]
        img = np.dstack((gasf_img,gadf_img,np.zeros(gadf_img.shape)))
    elif kind == "GASF_MTF":
        gasf_img = gasf.transform(pd.DataFrame(ts).T)[0]
        mtf_img = mtf.fit_transform(pd.DataFrame(ts).T)[0]
        
        img = np.dstack((gasf_img,mtf_img, np.zeros(gasf_img.shape)))
    elif kind == "GADF_MTF":
        gadf_img = gadf.transform(pd.DataFrame(ts).T)[0]
        mtf_img = mtf.fit_transform(pd.DataFrame(ts).T)[0]
        img = np.dstack((gadf_img,mtf_img, np.zeros(gadf_img.shape)))
    return img


# In[ ]:


def audio_mfcc(signal, sr, n_mfcc = 30):
    mfcc_signal = np.mean(librosa.feature.mfcc(y = signal, sr = sr, n_mfcc  =n_mfcc, fmin=300., fmax=600., center = True, n_mels = 20, n_fft = 1024), axis = 0)
    normalized_mfcc_feature = normalize(mfcc_signal)
    return np.array(normalized_mfcc_feature)

def convert_ts_to_images(DATASET_FILE, n_mfcc,  kind, res_sig_size, saveTo):
    pathlib.Path(saveTo).mkdir(parents=True, exist_ok=True)
    files = sorted(list(pathlib.Path(DATASET_FILE).rglob("*.wav")))
    for f in tqdm(files, total = len(files)):
        pathlib.Path(os.path.join(saveTo, f.parts[-2])).mkdir(parents=True, exist_ok=True)
        print("************")
        print(f)
        signal, sr = librosa.load(f, duration=6.5)
        print("**********##########")
#         signal = librosa.power_to_db(librosa.feature.melspectrogram(signal), top_db=None)
#         print(sr)
#         signal = signal[0:int(6.5*sr)]
#         print(len(signal))

        normalized_mfcc_feature = audio_mfcc(signal, sr, n_mfcc = n_mfcc)
        
        x = len(normalized_mfcc_feature) // res_sig_size
#         x = 0
        img = timeSeriesToImage(normalized_mfcc_feature, size_x =  None, kind = kind, window_size = x)
#         print(len(normalized_mfcc_feature), img.shape)
        
        cv2.imwrite(os.path.join(saveTo, f.parts[-2], f.stem + ".png"), img)
#     files1 = sorted(list(pathlib.Path(saveTo).rglob("*.wav")))
#     files2 = sorted(list(pathlib.Path(saveTo).rglob("*.png")))
#     print(len(files1), len(files2))
DATASET = "aug_dataset/"

for alg in tqdm(["GADF", "GASF", "MTF", "RP", "RGB_GAF", "GASF_MTF", "GADF_MTF"]):
    convert_ts_to_images("./aug_dataset/", 30, alg, 90, f"ALL_DATA/aug_{alg}_dataset_30_90/")


# In[ ]:




