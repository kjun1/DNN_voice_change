import torch
from torch import nn, optim
from sklearn.datasets import  load_digits

digits = load_digits()

X = digits.data
y = digits.target

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.int64)
#print(X.size(),y.size())
net = nn.Linear(X.size()[1],10)

loss_fn = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(),lr=0.01)

losses=[]

for epoc in range(100):
    optimizer.zero_grad()

    y_pred = net(X)
    print(y_pred.size(),y.size())
    loss = loss_fn(y_pred, y)
    loss.backward()

    optimizer.step()

    losses.append(loss.item())

_,y_pred = torch.max(net(X),1)

print((y_pred == y).sum().item()/len(y))
