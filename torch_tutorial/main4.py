import torch
from torch import nn,optim
from sklearn.datasets import  load_digits

digits = load_digits()

X = digits.data
y = digits.target

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.int64)




from torch.utils.data import TensorDataset, DataLoader

ds = TensorDataset(X,y)
loader = DataLoader(ds, batch_size=64, shuffle=True)

net = nn.Sequential(
    nn.Linear(64,32),
    nn.ReLU(),
    nn.Linear(32,16),
    nn.ReLU(),
    nn.Linear(16,10)
)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

losses = []
for epoch in range(10):
    running_loss = 0.0
    for xx,yy in loader:
        print(xx.size(),yy.size())
        y_pred = net(xx)
        #print(y_pred)
        loss = loss_fn(y_pred,yy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    losses.append(running_loss)
