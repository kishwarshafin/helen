import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import h5py
import torch


class SequenceDataset(Dataset):
    """
    Arguments:
        A CSV file path
    """

    def __init__(self, hdf5_path, transform=None):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.hdf5_file = h5py.File(hdf5_path, 'r')
        self.summary_keys = list(self.hdf5_file['summaries'].keys())

    def __getitem__(self, index):
        # load the image
        key = self.summary_keys[index]
        hdf5_file = self.hdf5_file['summaries'][key]

        image = hdf5_file['image']
        labels = hdf5_file['label']
        # chromosome_name = hdf5_file['chromosome_name']

        image = torch.Tensor(image)
        # label = torch.(image).type(torch.DoubleStorage)
        label = np.array(labels, dtype=np.int)
        # label = torch.from_numpy(label).type(torch.LongStorage)

        return image, label

    def __len__(self):
        return len(self.summary_keys)
