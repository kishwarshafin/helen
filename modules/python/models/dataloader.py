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
        hdf5_file = h5py.File(self.file_info[index], 'r')
        image = hdf5_file['image']
        label_base = hdf5_file['label_base']
        label_run_length = hdf5_file['label_run_length']
        # chromosome_name = hdf5_file['chromosome_name']

        image = torch.Tensor(image)
        # label = torch.(image).type(torch.DoubleStorage)
        label_base = np.array(label_base, dtype=np.int)
        label_run_length = np.array(label_run_length, dtype=np.int)
        # label = torch.from_numpy(label).type(torch.LongStorage)
        # A
        label_base = np.where(label_base == 65, 1, label_base)
        # C
        label_base = np.where(label_base == 67, 2, label_base)
        # G
        label_base = np.where(label_base == 71, 3, label_base)
        # T
        label_base = np.where(label_base == 84, 4, label_base)
        # GAP
        label_base = np.where(label_base == 95, 0, label_base)

        return image, label_base, label_run_length

    def __len__(self):
        return len(self.file_info)
