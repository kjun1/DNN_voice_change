import torch

w_true = torch.Tensor([1,2,3])

X = torch.cat([torch.ones(100,1),torch.randn(100,2)],1)

y = torch.mv(X, w_true) + torch.randn(100) * 0.5

w = torch.randn(3,requires_grad=True)


from torch import nn, optim

net = nn.Linear(in_features=3,out_features=1,bias=False)

optimizer = optim.SGD(net.parameters(), lr=0.1)

loss_fn = nn.MSELoss()


losses=[]
for epoc in range(100):

    optimizer.zero_grad()

    y_pred = net(X)

    loss = loss_fn(y_pred.view_as(y), y)
    loss.backward()

    optimizer.step()

    losses.append(loss.item())

print(list(net.parameters()))
