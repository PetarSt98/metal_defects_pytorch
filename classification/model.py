import config
from config import *


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.Conv1 = nn.Conv2d(3, 32, 3)
        self.Pool = nn.MaxPool2d(2)
        self.Conv2 = nn.Conv2d(32, 64, 3)
        self.Conv3 = nn.Conv2d(64, 128, 3)
        self.Flatten = nn.Flatten()
        self.Dense1 = nn.Linear(128 * 23 * 23, 256)
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
