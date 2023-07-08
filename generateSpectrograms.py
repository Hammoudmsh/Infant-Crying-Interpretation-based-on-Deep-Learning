import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import audioModule
import soundfile as sf

import os
import warnings
warnings.filterwarnings('ignore')
import wave

import random
import librosa
import numpy as np
import soundfile as sf
# install pydub for using HighPassFilter and play
from pydub.playback import play
from audiomentations import Compose, AddGaussianNoise, PitchShift, HighPassFilter
# import simpleaudio as sa
import matplotlib.pyplot as plt
#from helper import _plot_signal_and_augmented_signal
from IPython.display import Audio
import librosa.display as dsp
# import mir_eval
import pandas as pd
# import torch
# from torch.utils.data import Dataset, DataLoader
# import torchaudio
# from torchsummary import summary 
import os

import sounddevice as sd
from scipy.io.wavfile import write
import scipy.io.wavfile as wavfile
import wavio as wv

# from torch import nn
# from torchvision import datasets
# from torch.utils.tensorboard import SummaryWriter
# from torchvision.transforms import ToTensor
import pathlib
# import torchvision
# from tflite_model_maker import audio_classifier
# import tensorflow as ts
from tqdm.auto import tqdm

plt.rcParams["axes.labelsize"] = 'medium'
plt.rcParams["axes.titlecolor"] = 'red'
plt.rcParams["axes.titlesize"] = 'large'
#plt.rcParams["figure.figsize"] = (15, 10)
plt.rcParams["font.size"] = 18



import config




obj = audioModule.audioPreprocessing()





obj = audioModule.audioPreprocessing()

"""
from opensoundscape.audio import Audio
from opensoundscape.spectrogram import Spectrogram
def audio2Spectrogram(audio_filename, image_shape = (224, 224), duration = 7.0, sample_rate = 22050):
    # global obj
    # signal = obj.readAudio(audio_filename)
    # obj.plotSpectrum(signal, obj.sample_rate, 'log', "")
    # plt.axis('off');
    # plt.savefig(str(audio_filename)+".png")
    # image = cv2.imread("tmp.png")
    audio = Audio.from_file(audio_filename, sample_rate = sample_rate)
    spectrogram = Spectrogram.from_audio(audio)
    image = spectrogram.to_image(shape=image_shape, invert=False)
    return image


def generate_spectrogram1(audioFileNames):
    global obj
    # print("before: ", len(audioFileNames))
    for idx, x in tqdm(enumerate(audioFileNames), total=len(audioFileNames)):
        # fig, ax = plt.subplots(1, 1, figsize = (200, 200))
        # signal_org = obj.readAudio(x)
        # signal = obj._resample_if_necessary(signal_org, config.targetSampleRate)
        # signal = obj._mix_down_if_necessary(signal)
        # signal = obj._cut_if_necessary(signal, config.targetNumSamples)
        # signal = obj._right_pad_if_necessary(signal, config.targetNumSamples)

        image_save_path = os.path.join(
            pathlib.Path(x).parent,
            str(pathlib.Path(x).stem) + '.png'
            )
        print(image_save_path)
        # obj.plotSpectrum(signal_org, config.targetSampleRate, 'log', ax = ax)
        # fig.savefig("xsc.png")
        spectro  = audio2Spectrogram(x, (400, 400))
        spectro.save(image_save_path)
"""
def load_data(path):
    audioFiles = sorted(list(pathlib.Path(path).rglob("*.wav")))
    classes = [str(f.parent).split("\\")[-1] for f in audioFiles]
    return audioFiles, classes





def generate_spectrogram(audioFileNames):
    global obj
    # print("before: ", len(audioFileNames))
    for idx, x in tqdm(enumerate(audioFileNames), total=len(audioFileNames)):
        print(x)
        Fs, aud = wavfile.read(x)

        fig, ax = plt.subplots(1,1, tight_layout = True, frameon=False, figsize = (2.56,2.56))
        powerSpectrum, frequenciesFound, time, imageAxis = plt.specgram(aud, Fs=Fs)
        plt.axis('off')
        # plt.show()

        image_save_path = os.path.join(
            pathlib.Path(x).parent,
            str(pathlib.Path(x).stem) + '.png'
            )

        fig.savefig(image_save_path, bbox_inches = "tight")


if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    """ Load the data """
    # data_path = "last_dataset/"
    data_path = "aug_dataset/"

    X, y = load_data(data_path)
    print(f"Train: {len(X)} - {len(y)}")
    """ Soectro geneeration """
    generate_spectrogram(X)

