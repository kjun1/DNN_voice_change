from scipy.io import wavfile
from matplotlib import pyplot as plt
import pyworld as pw
import numpy as np
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import func
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
import torch.utils as utils



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


tsuchiya = func.wavedata(".")#tsuchiya = np.array(func.wavedata("tsuchiya_normal"))
uemura = func.wavedata(".")#uemura = np.array(func.wavedata("uemura_normal"))

X_train, X_test, y_train, y_test = train_test_split(
    tsuchiya, uemura, test_size=1/5, random_state=0)

X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test)
y_train = torch.Tensor(y_train)
y_test = torch.Tensor(y_test)

ds_train = TensorDataset(X_train, y_train)
ds_test = TensorDataset(X_test, y_test)

dataloader_train = DataLoader(ds_train,batch_size=16, shuffle=True)
dataloader_test = DataLoader(ds_test, shuffle=False)
#print(np.shape(spec))
#print(np.shape(f0))
print(ds_train[0])
"""
print(X_train.size(),y_test.size())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VAE(nn.Module):
    def __init__(self, z_dim):
      super(VAE, self).__init__()
      self.dense_enc1 = nn.Linear(1026, 200)
      self.dense_enc2 = nn.BatchNorm1d(200)
      self.dense_encmean = nn.Linear(200, z_dim)
      self.dense_encvar = nn.Linear(200, z_dim)

      self.dense_dec1 = nn.Linear(z_dim, 200)
      self.dense_dec2 = nn.Linear(200, 200)
      self.dense_dec3 = nn.Linear(200, 1026)

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


import numpy as np
from torch import optim

model = VAE(64).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
model.train()
for i in range(20):
  losses = []
  for x, t in dataloader_train:
      x = x.to(device)
      model.zero_grad()
      y = model(x)
      loss = model.loss(x)
      #print("loss is"+str(loss))
      loss.backward()
      optimizer.step()
      losses.append(loss.cpu().detach().numpy())
  print("EPOCH: {} loss: {}".format(i, np.average(losses)))



# 誤差関数の設定
loss_fn = nn.MSELoss()  # 変数名にはcriterionも使われる

# 重みを学習する際の最適化手法の選択
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(epoch):
    model.train()  # ネットワークを学習モードに切り替える

    # データローダーから1ミニバッチずつ取り出して計算する
    for data, target in loader_train:
        data, target = Variable(data), Variable(target)  # 微分可能に変換
        optimizer.zero_grad()  # 一度計算された勾配結果を0にリセット

        output = model(data)  # 入力dataをinputし、出力を求める
        #print(output.size())
        #print(target.size())
        loss = loss_fn(output, target)  # 出力と訓練データの正解との誤差を求める
        loss.backward()  # 誤差のバックプロパゲーションを求める
        optimizer.step()  # バックプロパゲーションの値で重みを更新する

    print("epoch{}：終了\n".format(epoch))

def test():
    model.eval()  # ネットワークを推論モードに切り替える
    correct = 0

    # データローダーから1ミニバッチずつ取り出して計算する
    for data, target in loader_test:
        data, target = Variable(data), Variable(target)  # 微分可能に変換
        print(data)
        output = model(data)  # 入力dataをinputし、出力を求める

        # 推論する
        pred = output.data.max(1, keepdim=True)[1]  # 出力ラベルを求める
        correct += pred.eq(target.data.view_as(pred)).sum()  # 正解と一緒だったらカウントアップ

    # 正解率を出力
    data_num = len(loader_test.dataset)  # データの総数
    print('\nテストデータの正解率: {}/{} ({:.0f}%)\n'.format(correct,
                                                   data_num, 100. * correct / data_num))
#[print(i) for i in loader_train]
for epoch in range(100):
    train(epoch)
"""
