from scipy.io import wavfile
import os
import numpy as np
import pyworld as pw

def db(x, dBref):
    y = 20 * np.log10(x / dBref)                      # リニア値をdB値に変換
    return y                                          # dB値を返す

def lin(x, dBref):
    y = dBref * 10 ** (x / 20)                        # dB値をリニア値に変換
    return y                                          # リニア値を返す


# wavディレクトリを指定するとspをarrayにして返す関数
def wavesp(path):
    WAVEsp = []
    WAV_FILE = [i for i in os.listdir(path=path) if i[-3:]=="wav"]
    for i, name in enumerate(WAV_FILE):
        print(i, name)
        fs, data = wavfile.read(path+"/"+name)
        data = data.astype(np.float)
        f0, t = pw.dio(data, fs)  # 基本周波数の抽出
        f0 = pw.stonemask(data, f0, t, fs)  # refinement
        sp = pw.cheaptrick(data, f0, t, fs)  # スペクトル包絡の抽出
        spec = db(sp, 2e-5)
        spec = np.concatenate([sp,np.zeros((3293-np.shape(spec)[0],1025))]) #長さをそろえるのは面倒なので数字で指定している
        f0 = np.array([np.concatenate([f0,np.zeros(3293-len(f0))])]).T
        #spec = lin(spec, 2e-5)
        x = np.concatenate([f0,spec],1)
        WAVEsp.append(x)
    WAVEsp = np.array(WAVEsp)
    return WAVEsp
