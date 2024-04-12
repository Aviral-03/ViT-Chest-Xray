import numpy as np

from tqdm import tqdm, trange

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor

# Dataset
def main():
  

# Vision Transformer
class MyViT(nn.Module):
  def __init__(self):
    super(MyViT, self).__init__()

  def forward(self, images):
    pass