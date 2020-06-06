from scipy.io import wavfile
from matplotlib import pyplot as plt
import pyworld as pw
import numpy as np
import os
import torch
import func

def db(x, dBref):
    y = 20 * np.log10(x / dBref)                      # リニア値をdB値に変換
    return y                                          # dB値を返す

def lin(x, dBref):
    y = dBref * 10 ** (x / 20)                        # dB値をリニア値に変換
    return y                                          # リニア値を返す

print("Please white the directory path")
path = input() # ディレクトリパスの読み込み
WAV_FILE = [i for i in os.listdir(path=path) if i[-3:]=="wav"] # wavfileのパスの読み込み

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

WAVEsp = []

for i, name in enumerate(WAV_FILE):
    print(i, name)
    fs, data = wavfile.read(path+"/"+name)
    data = data.astype(np.float)
    f0, t = pw.dio(data, fs)  # 基本周波数の抽出
    f0 = pw.stonemask(data, f0, t, fs)  # refinement
    sp = pw.cheaptrick(data, f0, t, fs)  # スペクトル包絡の抽出
    spec = db(sp, 2e-5)
    spec = np.concatenate([sp,np.zeros((3293-np.shape(spec)[0],1025))])
    f0 = np.array([np.concatenate([f0,np.zeros(3293-len(f0))])]).T
    #spec = lin(spec, 2e-5)
    x = np.concatenate([f0,spec],1)
    WAVEsp.append(x)
WAVEsp = np.array(WAVEsp)
X_train = torch.from_numpy(WAVEsp)
print(X_train)
#print(np.shape(spec))
#print(np.shape(f0))
