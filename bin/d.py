import pickle
import os
from scipy.io import wavfile
import pyworld as pw
import numpy as np
import pysptk

print("input name")
name = "uemura_normal"
path = "../../dataset/wav_data/"+name

wav_file = ["../../dataset/wav_data/"+name+"/"+i for i in os.listdir(path="../../dataset/wav_data/"+name) if i[-3:]=="wav"]
for i , n in enumerate(wav_file):
    fs, data = wavfile.read(n)
    data = data.astype(np.float)  # WORLDはfloat前提のコードになっているのでfloat型にしておく


    f0, t = pw.dio(data, fs)  # 基本周波数の抽出
    f0 = pw.stonemask(data, f0, t, fs)  # refinement
    sp = pw.cheaptrick(data, f0, t, fs)  # スペクトル包絡の抽出
    ap = pw.d4c(data, f0, t, fs)  # 非周期性指標の抽出)

    if i != 0:
        d = np.concatenate([d, sp], 0)
        fd = np.concatenate([fd, f0], 0)
    else:
        d = sp
        fd = f0
    
