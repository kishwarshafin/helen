import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import h5py
import torch
import pickle


class SequenceDataset(Dataset):
    """
    Arguments:
        A CSV file path
    """

    def __init__(self, csv_path, transform=None):
        data_frame = pd.read_csv(csv_path, header=None, dtype=str)
        self.transform = transform

        self.file_info = list(data_frame[0])

    def __getitem__(self, index):
        hdf5_filename = self.file_info[index]

        hdf5_file = h5py.File(hdf5_filename, 'r')
        chromosome_name = hdf5_filename.split('.')[-3].split('-')[0]

        image_dataset = hdf5_file['simpleWeight']
        position_dataset = hdf5_file['position']

        image = []
        for image_line in image_dataset:
            image.append(list(image_line))

        position = []
        index = []
        for pos, indx in position_dataset:
            position.append(pos)
            index.append(indx)

        hdf5_file.close()

        image = torch.Tensor(image)
        position = np.array(position, dtype=np.int)
        index = np.array(index, dtype=np.int)

        return image, chromosome_name, position, index

    def __len__(self):
        return len(self.file_info)
