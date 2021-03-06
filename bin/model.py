import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, z_dim, f_dim):
        super(VAE, self).__init__()
        self.dense_enc1 = nn.Linear(40, 256)
        self.dense_enc2 = nn.Linear(256,128)
        self.dense_encmean = nn.Linear(128, z_dim)
        self.dense_encvar = nn.Linear(128, z_dim)

        self.dense_dec1 = nn.Linear(z_dim+f_dim, 128)
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

    def _decoder(self, z, f):
        x = F.relu(self.dense_dec1(torch.cat([z,f], dim=1)))
        x = F.relu(self.dense_dec2(x))
        x = F.sigmoid(self.dense_dec3(x))
        return x

    def forward(self, x, f):
        mean, var = self._encoder(x)
        z = self._sample_z(mean, var)
        x = self._decoder(z, f)
        return x, z

    def loss(self, x, f):
        mean, var = self._encoder(x)
        delta = 1e-7
        KL = -0.5 * torch.mean(torch.sum(1 + torch.log(var + delta) - mean**2 - var))
        z = self._sample_z(mean, var)
        y = self._decoder(z, f)
        reconstruction = torch.mean(torch.sum(x * torch.log(y) + (1 - x) * torch.log(1 - y)))
        lower_bound = [-KL, reconstruction]
        return -sum(lower_bound)
