from scipy.io import wavfile
from matplotlib import pyplot as plt
import pyworld as pw
import numpy as np
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import func
from sklearn.model_selection import train_test_split



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


tsuchiya = np.array(func.wavedata("."))#tsuchiya = np.array(func.wavedata("tsuchiya_normal"))
uemura = np.array(func.wavedata("."))#uemura = np.array(func.wavedata("uemura_normal"))

X_train, X_test, y_train, y_test = train_test_split(
    tsuchiya, uemura, test_size=1/5, random_state=0)

X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test)
y_train = torch.Tensor(y_train)
y_test = torch.Tensor(y_test)

ds_train = TensorDataset(X_train, y_train)
ds_test = TensorDataset(X_test, y_test)

loader_train = DataLoader(ds_train, batch_size=64, shuffle=True)
loader_test = DataLoader(ds_test, shuffle=False)
#print(np.shape(spec))
#print(np.shape(f0))
