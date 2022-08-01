import PIL.Image
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
import cv2
from torch.utils.data.dataset import Dataset
from torchvision import transforms, datasets
import torch.optim as optim
from PIL import Image
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
TRAIN_DIR = './../dataset/train/'
VAL_DIR = './../dataset/valid/'
TEST_DIR = './../dataset/test/'
DATASET_DIR = './../dataset/'
EPOCHS = 30
LR = 1e-4
WD = 1e-6
BATCH_SIZE = 32
LABELS = ['Crazing', 'Inclusion', 'Patches', 'Pitted', 'Rolled', 'Scratches']
SAVE_MODEL_PATH = './'
LOAD_MODEL_PATH = './'

transformations = transforms.Compose([transforms.Resize((200, 200), Image.Resampling.NEAREST), transforms.ToTensor()])

print(DEVICE)
print(torch.__version__)
print('\n')
