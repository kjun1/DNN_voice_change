import torch
from torch import nn,optim
net = nn.Sequential(
    nn.Linear(64,32),
    nn.ReLU(),
    nn.Linear(32,16),
    nn.ReLU(),
    nn.Linear(16,10)
)

from sklearn.datasets import  load_digits

digits = load_digits()

X = digits.data
y = digits.target

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.int64)
print(X.size(),y.size())
loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters())

losses=[]

for epoc in range(100):
    optimizer.zero_grad()

    y_pred = net(X)

    loss = loss_fn(y_pred, y)
    loss.backward()

    optimizer.step()

    losses.append(loss.item())

_,y_pred = torch.max(net(X),1)

print((y_pred == y).sum().item()/len(y))
