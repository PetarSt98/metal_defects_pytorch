import config
from config import *


class VAE(torch.nn.Module):
    def __init__(self, in_channels, latent_dim=128, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 64, 128]

        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.conv_encoder = nn.Sequential(*modules)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 8 * 8, 128)
        self.fc_mu = nn.Linear(128, latent_dim)

        self.fc_log_var = nn.Linear(128, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 128 * 8 * 8, )
        self.relu = nn.ReLU()

        hidden_dims.reverse()
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(0.2))
            )
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1],
                                   3,
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1),
                nn.BatchNorm2d(3),
                nn.Sigmoid()
            )
        )
        self.conv_decoder = nn.Sequential(*modules)

    @staticmethod
    def reparametrize(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        sample = mu + (eps * std)
        return sample

    def forward(self, x):
        x = self.conv_encoder(x)
        x = self.flatten(x)
        hidden = self.fc1(x)
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)

        z = self.reparametrize(mu, log_var)
        z = self.fc2(z)
        z = z.view(z.shape[0], 128, 8, 8)

        reconstruction = self.conv_decoder(z)
        return reconstruction, mu, log_var

