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

loader_train = DataLoader(ds_train,batch_size=32, shuffle=True)
loader_test = DataLoader(ds_test, shuffle=False)
#print(np.shape(spec))
#print(np.shape(f0))

print(X_train.size(),y_test.size())

from torch import nn,optim
import func
from torch.autograd import Variable

model = nn.Sequential()
model.add_module('fc1', nn.Linear(1026, 100))
model.add_module('relu1', nn.ReLU())
model.add_module('fc2', nn.Linear(100, 100))
model.add_module('relu2', nn.ReLU())
model.add_module('fc3', nn.Linear(100, 1026))

print(model)

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
