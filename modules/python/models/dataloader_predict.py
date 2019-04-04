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

    def __init__(self, hdf5_path):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.hdf_filepath = hdf5_path
        hdf5_file = h5py.File(hdf5_path, 'r')
        self.total_summary_keys = len(list(hdf5_file['summaries'].keys()))
        hdf5_file.close()

    def __getitem__(self, index):
        # load the image
        hdf5_file_ref = h5py.File(self.hdf_filepath, 'r')
        key = list(hdf5_file_ref['summaries'].keys())[index]
        hdf5_file = hdf5_file_ref['summaries'][key]

        image = hdf5_file['image']
        chromosome = hdf5_file['chromosome_name'][()]
        position = hdf5_file['position']
        index = hdf5_file['index']
        # chromosome_name = hdf5_file['chromosome_name']

        image = torch.Tensor(image)
        # label = torch.(image).type(torch.DoubleStorage)
        position = np.array(position, dtype=np.int64)
        index = np.array(index, dtype=np.int)
        chromosome = str(chromosome)
        # label = torch.from_numpy(label).type(torch.LongStorage)
        hdf5_file_ref.close()

        return image, chromosome, position, index

    def __len__(self):
        return self.total_summary_keys
