import pickle
import os
from scipy.io import wavfile
import pyworld as pw
import numpy as np
import pysptk

print("input name")
name = "tsuchiya_normal"
path = "../../wav_data/"+name

wav_file = ["../../wav_data/"+name+"/"+i for i in os.listdir(path="../../wav_data/"+name) if i[-3:]=="wav"]
for i , name in enumerate(wav_file):
    fs, data = wavfile.read(name)
    data = data.astype(np.float)  # WORLDはfloat前提のコードになっているのでfloat型にしておく


    f0, t = pw.dio(data, fs)  # 基本周波数の抽出
    f0 = pw.stonemask(data, f0, t, fs)  # refinement
    sp = pw.cheaptrick(data, f0, t, fs)  # スペクトル包絡の抽出
    ap = pw.d4c(data, f0, t, fs)  # 非周期性指標の抽出)

    #[print(sp[i]) for i in range(100)]
    alpha = 0.46
    mcep = pysptk.sp2mc(sp, 39, alpha)
    #print(mcep.shape)
    #print(f0.shape)
    #a = np.block([mcep, f0.reshape(len(f0),1)])
    a = mcep
    if i != 0:
        d = np.concatenate([d, a], 0)
    else:
        d = a
    print(i)
    print(d.shape)



min = np.min(d)
d = d - min
max = np.max(d)
d = d/max

with open(name+".binaryfile", "wb") as web:
    pickle.dump([d,min,max], web)
