import glob
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import librosa
import librosa.display

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
    precision_recall_curve,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from sklearn.ensemble import RandomForestClassifier

from tqdm.notebook import tqdm
from joblib import Parallel, delayed

import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


DIR_BASE = '/Users/calvinhinkle/Desktop/school/mines/2026/spring/adML/finalProj/'
DIR_DRONE = os.path.join(DIR_BASE, 'Binary_Drone_Audio/yes_drone')
DIR_UNKNOWN = os.path.join(DIR_BASE, 'Binary_Drone_Audio/unknown')
DIR_TEST = os.path.join(DIR_BASE, 'live_samples')

DURATION = 1.024  #sec
SR = 16000

LENGTH_N = int(SR * DURATION)

N_FFT = 1024
HOP_LENGTH = N_FFT // 4

TEST_FRAC = 0.2
RANDOM_SEED = 69

#neural net
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
PIN_MEMORY = DEVICE == "cuda"
LEARNING_RATE = 2 ** (-16)
BATCH_SIZE = 2**8
MAX_NUM_EPOCHS = 2**8
TRAIN_CONVERGE_WINDOW = 2**3
TRAIN_CONVERGE_STD = 2 ** (-12)

#rf
NUM_ESTIMATORS = 100
MAX_DEPTH = 80
MIN_SAMPLES_LEAF = 4
MAX_FEATURES = 0.3
MIN_SAMPLES_SPLIT = 20

#audio processing
def fix_length(y, target_len):
    if len(y) > target_len:
        return y[:target_len]
    else:
        return np.pad(y, (0, target_len - len(y)))

def preprocess(path):
    y_raw, _ = librosa.load(path, mono=True, sr=SR) #NEED sr=SR

    # normalize amplitude
    y_norm = librosa.util.normalize(y_raw)

    # force fixed duration
    y = fix_length(y_norm, LENGTH_N)

    return y

#feature extraction

def extract_features(y, sr=SR):

    features = []
    
    S_complex = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S = np.abs(S_complex)

    mel_spec = librosa.feature.melspectrogram(S=S, sr=sr)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec), n_mfcc=20)

    #S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))

    #mfcc
    #mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))
    
    #time mfcc data
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    features.extend(np.mean(delta_mfcc, axis=1))
    features.extend(np.std(delta_mfcc, axis=1))

    features.extend(np.mean(delta2_mfcc, axis=1))
    features.extend(np.std(delta2_mfcc, axis=1))

    #spectral feats
    spec_centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
    spec_bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)
    spec_rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)
    spec_contrast = librosa.feature.spectral_contrast(S=S, sr=sr)

    for f in [spec_centroid, spec_bandwidth, spec_rolloff]:
        features.append(np.mean(f))
        features.append(np.std(f))

    features.extend(np.mean(spec_contrast, axis=1))
    features.extend(np.std(spec_contrast, axis=1))

    #harmonic, percussive
    #y_harm, y_perc = librosa.effects.hpss(y)
    #features.append(np.sum(y_harm ** 2))
    #features.append(np.sum(y_perc ** 2))
    #total energy --> less expensive than hpss
    features.append(np.sum(y ** 2))

    #zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(np.mean(zcr))
    features.append(np.std(zcr))

    return np.array(features, dtype=np.float32)

#.wav to features
def wav_to_data(filepath):

    data = preprocess(filepath)

    if data.ndim > 1:
        data = data.mean(axis=1)

    features = extract_features(data)

    return features

def process_file(filepath, label):
    try:
        features = wav_to_data(filepath)
        return features, label
    except Exception as e:
        print(f"Skipping {filepath}: {e}")
        return None

def load_dataset(dir_drone, dir_unknown, n_jobs=-1):
    tasks = []

    for label, folder in [(0, dir_drone), (1, dir_unknown)]:
        for filename in os.listdir(folder):
            if filename.lower().endswith(".wav"):
                filepath = os.path.join(folder, filename)
                tasks.append((filepath, label))

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(process_file)(fp, lbl) for fp, lbl in tasks
    )

    results = [r for r in results if r is not None]

    X, y = zip(*results)
    return np.stack(X), np.array(y, dtype=np.int64)

