import os
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import h5py
import pandas as pd
import torch


class SequenceDataset(Dataset):
    """
    Arguments:
        A CSV file path
    """

    def __init__(self, csv_path, transform=None):
        self.transform = transforms.Compose([transforms.ToTensor()])
        data_frame = pd.read_csv(csv_path, header=None, dtype=str)
        # assert data_frame[0].apply(lambda x: os.path.isfile(x.split(' ')[0])).all(), \
        #     "Some images referenced in the CSV file were not found"
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.file_info = list(data_frame[0])

    def __getitem__(self, index):
        # load the image
        with h5py.File(self.file_info[index], 'r') as hdf5_file:
            image = hdf5_file['image'][()]
            label_base = hdf5_file['label_base'][()]
            label_run_length = hdf5_file['label_run_length'][()]

        return image, label_base, label_run_length

    def __len__(self):
        return len(self.file_info)
