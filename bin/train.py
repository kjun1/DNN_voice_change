for model import VAE

device = torch.device("cpu")



model = VAE(64).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.train()
for i in range(20):
  losses = []
  for x, f in dataloader_train:
      x = x.to(device)
      f = f.to(device)
      model.zero_grad()
      y = model(x,f)
      loss = model.loss(x,f)
      loss.backward()
      optimizer.step()
      losses.append(loss.cpu().detach().numpy())
  print("EPOCH: {} loss: {}".format(i, np.average(losses)))
