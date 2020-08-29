from scipy.io import wavfile
import pyworld as pw
import numpy as np
import pysptk
from math import log10
fs, data = wavfile.read("tsuchiya_normal_001.wav")
data = data.astype(np.float)  # WORLDはfloat前提のコードになっているのでfloat型にしておく


f0, t = pw.dio(data, fs)  # 基本周波数の抽出
f0 = pw.stonemask(data, f0, t, fs)  # refinement
sp = pw.cheaptrick(data, f0, t, fs)  # スペクトル包絡の抽出
ap = pw.d4c(data, f0, t, fs)  # 非周期性指標の抽出)

#[print(sp[i]) for i in range(100)]
alpha = 0.46
mcep = pysptk.sp2mc(sp, 40, alpha)

sp_from_mcep = pysptk.mc2sp(mcep, alpha, fftlen = 2048)

synthesized = pw.synthesize(f0, sp, ap, fs)
wavfile.write('./world.wav',fs,synthesized.astype(np.int16)) #int16にしないと音割れする

syn_mcep = pw.synthesize(f0, sp_from_mcep, ap, fs)
wavfile.write('./world_mcep.wav',fs,syn_mcep.astype(np.int16)) #int16にしないと音割れする
