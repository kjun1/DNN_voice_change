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

fs, data = wavfile.read("tsuchiya_normal_001.wav")
data = data.astype(np.float)  # WORLDはfloat前提のコードになっているのでfloat型にしておく


f0, t = pw.dio(data, fs)  # 基本周波数の抽出
f0 = pw.stonemask(data, f0, t, fs)  # refinement
sp = pw.cheaptrick(data, f0, t, fs)  # スペクトル包絡の抽出
ap = pw.d4c(data, f0, t, fs)  # 非周期性指標の抽出)

#[print(sp[i]) for i in range(100)]
alpha = 0.46
mcep = pysptk.sp2mc(sp, 39, alpha)
#print(mcep.shape)
"""
for i in range(40):
    mcep[:, i] = zscore(mcep[:, i])
"""
X_train, X_test, y_train, y_test = train_test_split(
    mcep, mcep, test_size=1/5, random_state=0)

X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test)
y_train = torch.Tensor(y_train)
y_test = torch.Tensor(y_test)

ds_train = TensorDataset(X_train, y_train)
ds_test = TensorDataset(X_test, y_test)

dataloader_train = DataLoader(ds_train,batch_size=16, shuffle=True)
dataloader_test = DataLoader(ds_test, shuffle=False)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VAE(nn.Module):
    def __init__(self, z_dim):
      super(VAE, self).__init__()
      self.dense_enc1 = nn.Linear(40, 256)
      self.dense_enc2 = nn.Linear(256,128)
      self.dense_encmean = nn.Linear(128, z_dim)
      self.dense_encvar = nn.Linear(128, z_dim)

      self.dense_dec1 = nn.Linear(z_dim, 128)
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
      x = F.relu(self.dense_dec1(z))
      x = F.relu(self.dense_dec2(x))
      x = F.sigmoid(self.dense_dec3(x))
      return x

    def forward(self, x):
      mean, var = self._encoder(x)
      z = self._sample_z(mean, var)
      x = self._decoder(z)
      return x, z

    def loss(self, x):
      mean, var = self._encoder(x)
      delta = 1e-7
      KL = -0.5 * torch.mean(torch.sum(1 + torch.log(var + delta) - mean**2 - var))
      z = self._sample_z(mean, var)
      y = self._decoder(z)
      reconstruction = torch.mean(torch.sum(x * torch.log(y) + (1 - x) * torch.log(1 - y)))
      lower_bound = [-KL, reconstruction]
      return -sum(lower_bound)



print(VAE)



model = VAE(64).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
model.train()
for i in range(15):
  losses = []
  for x, t in dataloader_train:
      x = x.to(device)
      model.zero_grad()
      y = model(x)
      loss = model.loss(x)
      #print("loss is "+str(loss))
      loss.backward()
      optimizer.step()
      losses.append(loss.cpu().detach().numpy())
  print("EPOCH: {} loss: {}".format(i, np.average(losses)))


model.eval()  # ネットワークを推論モードに切り替える

fs, data = wavfile.read("tsuchiya_normal_001.wav")
data = data.astype(np.float)  # WORLDはfloat前提のコードになっているのでfloat型にしておく


f0, t = pw.dio(data, fs)  # 基本周波数の抽出
f0 = pw.stonemask(data, f0, t, fs)  # refinement
sp = pw.cheaptrick(data, f0, t, fs)  # スペクトル包絡の抽出
ap = pw.d4c(data, f0, t, fs)  # 非周期性指標の抽出)

#[print(sp[i]) for i in range(100)]
alpha = 0.46
mcep = pysptk.sp2mc(sp, 39, alpha)
#print(mcep.shape)
"""
for i in range(40):
    mcep[:, i] = zscore(mcep[:, i])
"""
# データローダーから1ミニバッチずつ取り出して計算する
for i in range(len(mcep[0])):
    mcep[i, :] = model(torch.Tensor(mcep[i, :]))[0].to('cpu').detach().numpy().copy()  # 入力dataをinputし、出力を求める


sp_from_mcep = pysptk.mc2sp(mcep, alpha, fftlen = 2048)

synthesized = pw.synthesize(f0, sp, ap, fs)
wavfile.write('./world.wav',fs,synthesized.astype(np.int16)) #int16にしないと音割れする

synthesized = pw.synthesize(f0, modifiedsp(sp,1.2,48000), ap, fs)
wavfile.write('./world_1.2.wav',fs,synthesized.astype(np.int16)) #int16にしないと音割れする

syn_mcep = pw.synthesize(f0, sp_from_mcep, ap, fs)
wavfile.write('./world_mcep.wav',fs,syn_mcep.astype(np.int16)) #int16にしないと音割れする
