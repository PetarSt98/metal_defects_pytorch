import PIL.Image
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
import cv2
import torch.utils.data
import torchvision
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from IPython.display import clear_output
from torch.utils.tensorboard import SummaryWriter

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
TRAIN_DIR = './../dataset/train/'
EPOCHS = 300
LR = 2e-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
Z_DIM = 128
GEN_EMBEDDING = 128
FEATURES_GEN = 32
FEATURES_DISC = 32
CHANNELS_NUM = 3
LAMBDA_GP = 10
BETA1 = 0.5
CRITIC_ITERATIONS = 5
LABELS = ['Crazing', 'Inclusion', 'Patches', 'Pitted', 'Rolled', 'Scratches']
SAVE_MODEL_PATH = './'
LOAD_MODEL_PATH = './'

transformations = transforms.Compose([transforms.ToTensor(), transforms.Resize((64, 64)), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

print(DEVICE)
print(torch.__version__)
print('\n')
