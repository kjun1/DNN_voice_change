from scipy.io import wavfile
from matplotlib import pyplot as plt
import pyworld as pw
import numpy as np
import os
import torch
import func




"""
print("Please choose wavfile number")
for i, name in enumerate(WAV_FILE):
    print(i, name)
choose_wav = int(input())

fs, data = wavfile.read(path+"/"+WAV_FILE[choose_wav])

data = data.astype(np.float)  # WORLDはfloat前提のコードになっているのでfloat型にしておく

f0, t = pw.dio(data, fs)  # 基本周波数の抽出
f0 = pw.stonemask(data, f0, t, fs)  # refinement
sp = pw.cheaptrick(data, f0, t, fs)  # スペクトル包絡の抽出
ap = pw.d4c(data, f0, t, fs)  # 非周期性指標の抽出
"""


WAVEsp = np.array(func.wavesp("."))
X_train = torch.from_numpy(WAVEsp)
print(X_train)
#print(np.shape(spec))
#print(np.shape(f0))
