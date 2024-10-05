import torch
from torch.utils.data import Dataset
from torchvision import transforms as tt
from monai import transforms as mT ## Breaks with numpy > 2.0
from monai.utils import set_determinism

import os
from dotenv import load_dotenv
from pathlib import PosixPath, Path
import json
import numpy as np
import yaml
from typing import List, Dict, Tuple, Optional, Union, Any
from tqdm.notebook import tqdm

## Define all relevant transforms!
OUT_CHANNELS = 6
TRAIN_TRANSFORMS =  mT.Compose([
    mT.LoadImage(image_only=True),
    mT.EnsureChannelFirst(), ## Add a channel to the batch dimension
    mT.ScaleIntensity(),
    mT.RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
    mT.RandFlip(spatial_axis=0, prob=0.5),
    mT.RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
    mT.ToTensor(),
    ])

FTUNE_TRANSFORMS = mT.Compose([
    mT.LoadImage(image_only=True),
    mT.EnsureChannelFirst(), ## Add a channel to the batch dimension
    mT.ScaleIntensity(),
])

## Dataset!
class MedNIST_Dataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            data_dict: Dict, 
            transforms: mT.Compose, 
            image_key: str = "image",
            label_key: str = "label",
            ) -> None:
        self.data = data_dict
        self.transform = transforms
        self.image_key = image_key
        self.label_key = label_key

    def __len__(self):
        return len(self.data[self.image_key])
    
    def __getitem__(self, index):
        return {
            "x": self.transform(self.data[self.image_key][index]),
            "y": int(self.data[self.label_key][index]),}
