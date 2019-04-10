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

    def __init__(self, csv_path, transform=None):
        # self.transform = transforms.Compose([transforms.ToTensor()])
        data_frame = pd.read_csv(csv_path, header=None, dtype=str)
        # assert data_frame[0].apply(lambda x: os.path.isfile(x.split(' ')[0])).all(), \
        #     "Some images referenced in the CSV file were not found"
        # self.transform = transforms.Compose([transforms.ToTensor()])
        self.file_info = list(data_frame[0])

    def __getitem__(self, index):
        file_name = self.file_info[index].split('/')[-1]
        # load the image
        hdf5_file = h5py.File(self.file_info[index], 'r')
        # this needs to change
        contig = file_name.split('.')[-3].split('-')[0]

        contig_start = hdf5_file['contig_start'][0]
        contig_end = hdf5_file['contig_end'][0]
        chunk_id = hdf5_file['feature_chunk_idx'][0]
        image = hdf5_file['image']
        position = hdf5_file['position']

        image = torch.Tensor(image)
        position = np.array(position, dtype=np.int)

        return contig, contig_start, contig_end, chunk_id, image, position

    def __len__(self):
        return len(self.file_info)
