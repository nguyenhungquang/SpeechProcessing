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
DATA_ROOT=os.getcwd()

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
else: 
    with open("class.pkl", "rb") as file: 
        gmm=pickle.load(file)
for k in range(3):
    plt.plot(gmm.means_[k], linewidth=1.5, label="Mean of mixture {}".format(k+1))
plt.show()
plt.imshow(gmm.covariances_[0], origin="bottom left")
plt.show()
for k in range(3):
    plt.plot(np.diag(gmm.covariances_[k]), linewidth=1.5,
         label="Diagonal part of covariance matrix, mixture {}".format(k))
plt.show()
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
class MyFileDataSource(CMUArcticWavFileDataSource):
    def __init__(self, *args, **kwargs):
        super(MyFileDataSource, self).__init__(*args, **kwargs)
        self.test_paths = None

    def collect_files(self):
        paths = super(
            MyFileDataSource, self).collect_files()
        paths_train, paths_test = train_test_split(
            paths, test_size=test_size, random_state=1234)

        # keep paths for later testing
        self.test_paths = paths_test

        return paths_train

    def collect_features(self, path):
        fs, x = wavfile.read(path)
        x = x.astype(np.float64)
        x=x.sum(axis=1)/2
        f0, timeaxis = pyworld.dio(x, fs, frame_period=frame_period)
        f0 = pyworld.stonemask(x, f0, timeaxis, fs)
        spectrogram = pyworld.cheaptrick(x, f0, timeaxis, fs)
        spectrogram = trim_zeros_frames(spectrogram)
        mc = pysptk.sp2mc(spectrogram, order=order, alpha=alpha)
        return mc
clb_source = MyFileDataSource(data_root=DATA_ROOT,
                                         speakers=["clb"], max_files=max_files)
slt_source = MyFileDataSource(data_root=DATA_ROOT,
                                         speakers=["slt"], max_files=max_files)

X = PaddedFileSourceDataset(clb_source, 5000).asarray()
Y = PaddedFileSourceDataset(slt_source, 5000).asarray()
X_aligned, Y_aligned = DTWAligner(verbose=0, dist=melcd).transform((X, Y))
X_aligned, Y_aligned = X_aligned[:, :, 1:], Y_aligned[:, :, 1:]
static_dim = X_aligned.shape[-1]
if use_delta:
    X_aligned = apply_each2d_trim(delta_features, X_aligned, windows)
    Y_aligned = apply_each2d_trim(delta_features, Y_aligned, windows)
def vis_difference(x, y, which_dims=[0,2,3,6,8], T_max=None):
    static_paramgen = MLPG(gmm, windows=[(0,0, np.array([1.0]))], diff=False)
    paramgen = MLPG(gmm, windows=windows, diff=False)

    x = trim_zeros_frames(x)
    y = trim_zeros_frames(y)[:,:static_dim]
    y_hat1 = static_paramgen.transform(x)[:,:static_dim]
    y_hat2 = paramgen.transform(x)

    if T_max is not None and len(y) > T_max:
        y,y_hat1,y_hat2 = y[:T_max],y_hat1[:T_max],y_hat2[:T_max]

    plt.figure(figsize=(16,4*len(which_dims)))
    for idx, which_dim in enumerate(which_dims):
        plt.subplot(len(which_dims), 1, idx+1)
        plt.plot(y[:,which_dim], "--", linewidth=1, label="Target")
        plt.plot(y_hat1[:,which_dim], "-", linewidth=2, label="w/o MLPG")
        plt.plot(y_hat2[:,which_dim], "-", linewidth=3, label="w/ MLPG")
        plt.title("{}-th coef".format(which_dim+1), fontsize=16)
        #legend(prop={"size": 16}, loc="upper right")
    plt.show()
idx = 0
which_dims = np.arange(0, static_dim, step=2)
vis_difference(X_aligned[idx], Y_aligned[idx], T_max=300)#, which_dims=which_dims)
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
