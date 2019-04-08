import os
import numpy as np
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
        label_base = hdf5_file['label_base']
        label_rle = hdf5_file['label_rle']
        # chromosome_name = hdf5_file['chromosome_name']

        image = torch.Tensor(image)
        # label = torch.(image).type(torch.DoubleStorage)
        label_base = np.array(label_base, dtype=np.int)
        label_rle = np.array(label_rle, dtype=np.int)
        # label = torch.from_numpy(label).type(torch.LongStorage)
        hdf5_file_ref.close()

        return image, label_base, label_rle

    def __len__(self):
        return self.total_summary_keys
