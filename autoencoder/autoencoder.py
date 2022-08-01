import config
from config import *


class Autoencoder (nn.Module):
    def __init__(self):
        super().__init__()

        self.Encode1 = nn.Conv2d(3, 64, 3, padding=1)
       # nn.init.xavier_normal(self.Encode1.weight)
        self.Pool1 = nn.MaxPool2d(2, return_indices=True)
        self.Encode2 = nn.Conv2d(64, 128, 3, padding=1)
       # nn.init.xavier_normal(self.Encode2.weight)
        self.Pool2 = nn.MaxPool2d(2, return_indices=True)
        self.FlattenE1 = nn.Flatten()
        self.Encode_lin = nn.Linear(128 * config.IMAGE_SIZE//4 * config.IMAGE_SIZE//4, config.LATENT_DIM)

        self.Decode_lin = nn.Linear(config.LATENT_DIM, 128 * config.IMAGE_SIZE//4 * config.IMAGE_SIZE//4)
        self.UnflattenD1 = nn.Unflatten(-1, (128, config.IMAGE_SIZE//4, config.IMAGE_SIZE//4))
        self.Decode1 = nn.ConvTranspose2d(128, 64, 3, padding=1)
       # nn.init.xavier_normal(self.Decode1.weight)
        self.Unpool1 = nn.MaxUnpool2d(2)
        self.Decode2 = nn.ConvTranspose2d(64, 3, 3, padding=1)
       # nn.init.xavier_normal(self.Decode2.weight)
        self.Unpool2 = nn.MaxUnpool2d(2)
        self.ReLU = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encoder(self, x):
        x = self.Encode1(x)
        x = self.ReLU(x)
        x, ind1 = self.Pool1(x)
        x = self.Encode2(x)
        x = self.ReLU(x)
        x, ind2 = self.Pool2(x)
        x = self.FlattenE1(x)
        x = self.Encode_lin(x)
        return x, ind1, ind2

    def decoder(self, x, ind1, ind2):
        x = self.Decode_lin(x)
        x = self.ReLU(x)
        x = self.UnflattenD1(x)
        x = self.Unpool1(x, ind1)
        x = self.Decode1(x)
        x = self.ReLU(x)
        x = self.Unpool2(x, ind2)
        x = self.Decode2(x)
        x = self.sigmoid(x)
        return x

    def forward(self, x):
        x, ind1, ind2 = self.encoder(x)
        x = self.decoder(x, ind2, ind1)
        return x


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def add_noise(x):
    return x + torch.normal(0, config.SIGMA, size=(config.BATCH_SIZE, config.LATENT_DIM)).to(config.DEVICE)
