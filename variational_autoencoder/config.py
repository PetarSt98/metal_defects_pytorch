import PIL.Image
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
import cv2
import torch.utils.data
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from IPython.display import clear_output

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
TRAIN_DIR = './../dataset/train/'
EPOCHS = 30
LR = 2e-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
LATENT_DIM = 128
SIGMA = 1
LABELS = ['Crazing', 'Inclusion', 'Patches', 'Pitted', 'Rolled', 'Scratches']
SAVE_MODEL_PATH = './'
LOAD_MODEL_PATH = './'

transformations = transforms.Compose([transforms.ToTensor()
                                      ,transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])#

print(DEVICE)
print(torch.__version__)
print('\n')
