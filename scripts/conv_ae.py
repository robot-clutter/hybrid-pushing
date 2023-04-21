import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt

default_params = {
    'encoder': {
        'filters': [16, 32, 64, 128],
        'kernels': [4, 4, 4, 4],
        'strides': [2, 2, 2, 2],
        'padding': [1, 1, 1, 1],
    },
    'decoder': {
        'filters': [128, 64, 32, 16],
        'kernels': [4, 4, 4, 4],
        'stride': [2, 2, 2, 2],
        'padding': [1, 1, 1, 1],
        'output_channels': 1  # use 1 for regression, 5 for classification
    },
    'regression': True
}


class Encoder(nn.Module):
    def __init__(self, latent_dim, params=default_params):
        super(Encoder, self).__init__()

        filters = params['encoder']['filters']
        kernels = params['encoder']['kernels']
        stride = params['encoder']['strides']
        pad = params['encoder']['padding']

        self.conv1 = nn.Conv2d(1, filters[0], kernels[0], stride=stride[0], padding=pad[0])
        self.conv2 = nn.Conv2d(filters[0], filters[1], kernels[1], stride=stride[1], padding=pad[1])
        self.conv3 = nn.Conv2d(filters[1], filters[2], kernels[2], stride=stride[2], padding=pad[2])
        self.conv4 = nn.Conv2d(filters[2], filters[3], kernels[3], stride=stride[3], padding=pad[3])

        self.fc = nn.Linear(8192, latent_dim)
        self.mu = nn.Linear(8192, latent_dim)
        self.logvar = nn.Linear(8192, latent_dim)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(self.conv4(x))
        x = x.view(-1, 8192)
        return nn.functional.relu(self.fc(x))


class Decoder(nn.Module):
    def __init__(self, latent_dim, params=default_params):
        super(Decoder, self).__init__()

        filters = params['decoder']['filters']
        kernels = params['decoder']['kernels']
        stride = params['decoder']['stride']
        pad = params['decoder']['padding']
        channels = params['decoder']['output_channels']

        self.regression = params['regression']

        self.latent = nn.Linear(latent_dim, 8192)

        self.deconv1 = nn.ConvTranspose2d(filters[0], filters[1], kernels[0], stride=stride[0], padding=pad[0])
        self.deconv2 = nn.ConvTranspose2d(filters[1], filters[2], kernels[1], stride=stride[1], padding=pad[1])
        self.deconv3 = nn.ConvTranspose2d(filters[2], filters[3], kernels[2], stride=stride[2], padding=pad[2])
        self.out = nn.ConvTranspose2d(filters[-1], channels, kernels[-1], stride=stride[-1], padding=pad[-1])

    def forward(self, x):
        x = self.latent(x)
        n = x.shape[0]
        x = x.view(n, 128, 8, 8)
        x = nn.functional.relu(self.deconv1(x))
        x = nn.functional.relu(self.deconv2(x))
        x = nn.functional.relu(self.deconv3(x))
        if self.regression:
            out = torch.sigmoid(self.out(x))
        else:
            out = self.out(x)
        return out


class ConvAe(nn.Module):
    def __init__(self, latent_dim, device='cuda', params=default_params):
        super().__init__()
        self.params = params
        self.latent_dim = latent_dim

        self.device = device
        self.encoder = Encoder(latent_dim, params=params).to(self.device)
        self.decoder = Decoder(latent_dim, params=params).to(self.device)

    def forward(self, x):
        x = self.encoder(x)
        rec_x = self.decoder(x)
        return rec_x


def plot_rec(ae, img, classification=False, device='cuda'):
    fig, ax = plt.subplots(1, 2)
    x = torch.FloatTensor(np.expand_dims(img, axis=[0, 1])).to(device)

    if classification:
        y = ae(x).cpu().detach().cpu().numpy().squeeze()
        rec = np.zeros((128, 128))
        for k in range(128):
            for j in range(128):
                rec[k, j] = np.argmax(y[:, k, j])
    else:
        rec = ae(x).detach().cpu().numpy().squeeze()

    ax[0].imshow(img)
    ax[1].imshow(rec)
    plt.show()