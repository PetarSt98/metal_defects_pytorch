import config
from config import *


class Model2(nn.Module):
    def __init__(self, in_channels, hidden_dims=None):
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
                    nn.ReLU())
            )
            in_channels = h_dim
        self.pos_e_layer = nn.Sequential(
                    nn.Conv2d(128, out_channels=256*1,
                              kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256*1),
                    nn.ReLU()
        )
        self.conv_encoder = nn.Sequential(*modules)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 8 * 8, 6)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv_encoder(x)
        x = self.pos_e_layer(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return self.softmax(x)

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.Conv1 = nn.Conv2d(3, 32, 3)
        self.Pool = nn.MaxPool2d(2)
        self.Conv2 = nn.Conv2d(32, 64, 3)
        self.Conv3 = nn.Conv2d(64, 128, 3)
        self.Flatten = nn.Flatten()
        self.Dense1 = nn.Linear(128 * 6 * 6, 256)
        self.Dropout = nn.Dropout(0.2)
        self.Dense2 = nn.Linear(256, 6)
        self.ReLU = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.Conv1(x)
        x = self.ReLU(x)
        x = self.Pool(x)
        x = self.Conv2(x)
        x = self.ReLU(x)
        x = self.Pool(x)
        x = self.Conv3(x)
        x = self.ReLU(x)
        x = self.Pool(x)
        x = self.Flatten(x)
        x = self.Dense1(x)
        x = self.ReLU(x)
        x = self.Dropout(x)
        x = self.Dense2(x)
        x = self.softmax(x)
        return x
