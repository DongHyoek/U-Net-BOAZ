import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class Custom_Dataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        lst_data = os.listdir(self.data_dir)

        lst_label = [f for f in lst_data if f.startswith('label')]
        lst_input = [f for f in lst_data if f.startswith('input')]

        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):
        label = Image.open(os.path.join(self.data_dir, self.lst_label[index]))
        input = Image.open(os.path.join(self.data_dir, self.lst_input[index]))

        dataset = {'input': input, 'label': label}

        if self.transform:
            data = self.transform(dataset['input'])
            label = self.transform(dataset['label']).long()

        return data,label