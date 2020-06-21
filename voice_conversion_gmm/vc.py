
from nnmnkwii.datasets import PaddedFileSourceDataset
from nnmnkwii.datasets.cmu_arctic import CMUArcticWavFileDataSource
from nnmnkwii.preprocessing.alignment import DTWAligner
from nnmnkwii.preprocessing import trim_zeros_frames, remove_zeros_frames, delta_features
from nnmnkwii.util import apply_each2d_trim
from nnmnkwii.metrics import melcd
from nnmnkwii.baseline.gmm import MLPG

from os.path import basename, splitext
import os
import sys
import time

import numpy as np
from scipy.io import wavfile
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
import pyworld
import pysptk
from pysptk.synthesis import MLSADF, Synthesizer
import librosa
import librosa.display
import IPython
from IPython.display import Audio
import matplotlib.pyplot as plt
import pickle
import argparse

parser=argparse.ArgumentParser()
parser.add_argument("-f","--file",help="input file name")
parser.add_argument("-fd","--folder",help="input folder name")
parser.add_argument("-m","--model",help="model file")
args=parser.parse_args()
fs =44100
fftlen = pyworld.get_cheaptrick_fft_size(fs)
alpha = pysptk.util.mcepalpha(fs)
order = 24
frame_period = 5
hop_length = int(fs * (frame_period * 0.001))
max_files = 100 # number of utterances to be used.
test_size = 0.03
use_delta = True

if use_delta:
    windows = [
        (0, 0, np.array([1.0])),
        (1, 1, np.array([-0.5, 0.0, 0.5])),
        (1, 1, np.array([1.0, -2.0, 1.0])),
    ]
else:
    windows = [
        (0, 0, np.array([1.0])),
    ]
if args.model:
    with open(args.model, "rb") as file: 
        gmm=pickle.load(file)
#legend(prop={"size": 16})
def test_one_utt(src_path, disable_mlpg=False, diffvc=True):
    # GMM-based parameter generation is provided by the library in `baseline` module
    if disable_mlpg:
        # Force disable MLPG
        paramgen = MLPG(gmm, windows=[(0,0, np.array([1.0]))], diff=diffvc)
    else:
        paramgen = MLPG(gmm, windows=windows, diff=diffvc)

    fs, x = wavfile.read(src_path)
    print(x)
    x = x.astype(np.float64)
    if len(x.shape)==2:
        x=x.sum(axis=1)/2
    f0, timeaxis = pyworld.dio(x, fs, frame_period=frame_period)
    f0 = pyworld.stonemask(x, f0, timeaxis, fs)
    spectrogram = pyworld.cheaptrick(x, f0, timeaxis, fs)
    aperiodicity = pyworld.d4c(x, f0, timeaxis, fs)

    mc = pysptk.sp2mc(spectrogram, order=order, alpha=alpha)
    c0, mc = mc[:, 0], mc[:, 1:]
    if use_delta:
        mc = delta_features(mc, windows)
    mc = paramgen.transform(mc)
    #if disable_mlpg and mc.shape[-1] != static_dim:
    #    mc = mc[:,:static_dim]
    #assert mc.shape[-1] == static_dim
    mc = np.hstack((c0[:, None], mc))
    if diffvc:
        mc[:, 0] = 0 # remove power coefficients
        engine = Synthesizer(MLSADF(order=order, alpha=alpha), hopsize=hop_length)
        b = pysptk.mc2b(mc.astype(np.float64), alpha=alpha)
        waveform = engine.synthesis(x, b)
    else:
        spectrogram = pysptk.mc2sp(
            mc.astype(np.float64), alpha=alpha, fftlen=fftlen)
        waveform = pyworld.synthesize(
            f0, spectrogram, aperiodicity, fs, frame_period)
        
    return waveform
src_path="src/5.wav"
tgt_path="tgt/5.wav"
if args.file:
    src_path="src/"+args.file
    tgt_path="tgt/"+args.file
    w_MLPG = test_one_utt(src_path, disable_mlpg=False)
    wavfile.write(tgt_path,fs,w_MLPG)
if args.folder:
    folder=args.folder
    tgt_folder="tgt_"+folder
    if not os.path.exists(tgt_folder):
        os.makedirs(tgt_folder)
    for n,f in enumerate(os.listdir(folder)):
        file_path=folder+"/"+f
        out_path=tgt_folder+'/'+f
        print(out_path)
        wave=test_one_utt(file_path,disable_mlpg=False)
        wavfile.write(out_path,fs,wave)

