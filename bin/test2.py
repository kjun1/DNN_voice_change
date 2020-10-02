from scipy.io import wavfile
from scipy.stats import zscore
from scipy import interpolate
import pyworld as pw
import numpy as np
import pysptk
from math import log10
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle

def modifiedsp(sp,sp_rate,fs): # スペクトル包絡の変換
    fft_size = (len(sp[1])-1)*2 # こいつ何？
    w = [i*fs/fft_size for i in range(fft_size)]
    w2 = [i*sp_rate for i in w] # wに伸縮率をかける
    mod_sp = np.zeros_like(sp)

    for i in range(np.shape(sp)[0]):
        tmp = np.log(sp[i, :]) # spのi行をlogにする
        tmp = np.append(tmp[:], tmp[len(tmp)-1:1:-1],axis=0) # tmpとreverse_tmpを繋げる
        fnc = interpolate.PchipInterpolator(w,tmp) # 線形補間?
        tmp2 = fnc(w2) #　さっき作った関数にw2をぶち込む
        mod_sp[i] = (np.exp(tmp2[:int(fft_size/2+1)])) # さっき変換したspの一部(tmp2)を新しいspにぶち込む

    return mod_sp
"""
wav_file = ["../wav_data/tsuchiya_normal/"+i for i in os.listdir(path="../wav_data/tsuchiya_normal") if i[-3:]=="wav"]
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

    if i == 5:
        break


min = np.min(d)
d = d - min
max = np.max(d)
d = d/max

with open("aaa.binaryfile", "wb") as web:
    pickle.dump([d,min,max], web)
"""
with open("tsuchiya_normal.binaryfile", "rb") as web:
    d,min,max,state = pickle.load(web)

"""
for i in range(40):
    mcep[:, i] = zscore(mcep[:, i])
"""
p = [[state,abs(1-state)]] * len(d)

X_train, X_test = train_test_split(
    d, test_size=1/5, random_state=0) #random_stateは乱数シードの固定
y_train, y_test = train_test_split(
    p, test_size=1/5, random_state=0) #random_stateは乱数シードの固定

X_train = torch.Tensor(X_train) # Tensorにする
X_test = torch.Tensor(X_test) # 上同様
y_train = torch.Tensor(y_train) # Tensorにする
y_test = torch.Tensor(y_test) # 上同様
ds_train = TensorDataset(X_train, y_train) # 入力データと教師データをまとめる(VAEなので同じデータを固めてる)
ds_test = TensorDataset(X_test, y_test) # 上同様
#print(ds_train[0][0])

dataloader_train = DataLoader(ds_train,batch_size=1024, shuffle=True)
dataloader_test = DataLoader(ds_test, shuffle=False)




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VAE(nn.Module):
    def __init__(self, z_dim):
      super(VAE, self).__init__()
      self.dense_enc1 = nn.Linear(40, 256)
      self.dense_enc2 = nn.Linear(256,128)
      self.dense_encmean = nn.Linear(128, z_dim)
      self.dense_encvar = nn.Linear(128, z_dim)

      self.dense_dec1 = nn.Linear(z_dim+2, 128)
      self.dense_dec2 = nn.Linear(128, 256)
      self.dense_dec3 = nn.Linear(256, 40)

    def _encoder(self, x):
      x = F.relu(self.dense_enc1(x))
      x = F.relu(self.dense_enc2(x))
      mean = self.dense_encmean(x)
      var = F.softplus(self.dense_encvar(x))
      return mean, var

    def _sample_z(self, mean, var):
      epsilon = torch.randn(mean.shape).to(device)
      return mean + torch.sqrt(var) * epsilon

    def _decoder(self, z):
      p = torch.Tensor([[1,0]]*len(z))　# 話者情報入れる部分
      #print(p.shape)
      z = torch.cat([z,p], dim=1)
      #print(z)
      x = F.relu(self.dense_dec1(z))
      x = F.relu(self.dense_dec2(x))
      x = F.sigmoid(self.dense_dec3(x))
      return x

    def forward(self, x, state):
      mean, var = self._encoder(x)
      z = self._sample_z(mean, var)
      x = self._decoder(z)
      return x, z

    def loss(self, x, state):
      mean, var = self._encoder(x)
      delta = 1e-7
      KL = -0.5 * torch.mean(torch.sum(1 + torch.log(var + delta) - mean**2 - var))
      #print(KL)
      z = self._sample_z(mean, var)
      y = self._decoder(z)
      #reconstruction = torch.mean(torch.sum((x-y)**2))
      reconstruction = torch.mean(torch.sum(x * torch.log(y) + (1 - x) * torch.log(1 - y)))
      #print(reconstruction)
      lower_bound = [-KL, reconstruction]
      return -sum(lower_bound)



print(VAE)



model = VAE(64).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.train()
for i in range(20):
  losses = []
  for x, t in dataloader_train:
      #print(x,t)
      x = x.to(device)
      model.zero_grad()
      y = model(x,0)
      loss = model.loss(x,0)
      #print("loss is "+str(loss))
      loss.backward()
      optimizer.step()
      losses.append(loss.cpu().detach().numpy())
  print("EPOCH: {} loss: {}".format(i, np.average(losses)))


model.eval()  # ネットワークを推論モードに切り替える

fs, data = wavfile.read("../wav_data/uemura_normal/uemura_normal_001.wav")
data = data.astype(np.float)  # WORLDはfloat前提のコードになっているのでfloat型にしておく


f0, t = pw.dio(data, fs)  # 基本周波数の抽出
f0 = pw.stonemask(data, f0, t, fs)  # refinement
sp = pw.cheaptrick(data, f0, t, fs)  # スペクトル包絡の抽出
ap = pw.d4c(data, f0, t, fs)  # 非周期性指標の抽出)

#[print(sp[i]) for i in range(100)]
alpha = 0.46
mcep = pysptk.sp2mc(sp, 39, alpha)
mcep = (mcep - min)/max
"""
#print(mcep.shape)
d = np.block([mcep, f0.reshape(len(f0),1)])

for i in range(40):
    mcep[:, i] = zscore(mcep[:, i])

# データローダーから1ミニバッチずつ取り出して計算する
for i in range(len(mcep[0])):
    d[i, :] = model(torch.Tensor(d[i, :]))[0].to('cpu').detach().numpy().copy()  # 入力dataをinputし、出力を求める

mcep = d[:,:-1]
f1 = d[:,-1]
for i in range(len(f0)):
    print(f0[i],f1[i])
"""
for i in range(len(mcep[0])):
    mcep[i, :] = model(torch.Tensor(mcep[i, :]))[0].to('cpu').detach().numpy().copy()  # 入力dataをinputし、出力を求める

sp_from_mcep = pysptk.mc2sp(mcep*max+min, alpha, fftlen = 2048)
syn_mcep = pw.synthesize(f0, sp_from_mcep, ap, fs)
wavfile.write('./kakunin.wav',fs,syn_mcep.astype(np.int16)) #int16にしないと音割れする

"""
sp_from_mcep = pysptk.mc2sp(mcep, alpha, fftlen = 2048)

synthesized = pw.synthesize(f0, sp, ap, fs)
wavfile.write('./world.wav',fs,synthesized.astype(np.int16)) #int16にしないと音割れする

synthesized = pw.synthesize(f0, modifiedsp(sp,1.2,48000), ap, fs)
wavfile.write('./world_1.2.wav',fs,synthesized.astype(np.int16)) #int16にしないと音割れする

syn_mcep = pw.synthesize(f0, sp_from_mcep, ap, fs)
wavfile.write('./world_mcep.wav',fs,syn_mcep.astype(np.int16)) #int16にしないと音割れする
"""
