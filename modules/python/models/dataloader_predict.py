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

        contig_start = hdf5_file['contig_start'][0].astype(np.int)
        contig_end = hdf5_file['contig_end'][0].astype(np.int)
        chunk_id = hdf5_file['feature_chunk_idx'][0].astype(np.int)
        image = hdf5_file['image']
        rle_predictions = hdf5_file['bayesian_run_length_prediction']
        normalization = hdf5_file['normalization']
        position = hdf5_file['position']

        rle_predictions = torch.Tensor(rle_predictions).view(-1, 1)
        image = torch.Tensor(image)
        normalization = torch.Tensor(normalization)
        image = torch.cat((rle_predictions, normalization, image), 1)

        if image.size()[0] != 1000 or image.size()[1] != 404:
            raise ValueError("INVALID IMAGE SIZE: " + str(image.size()) + " " + str(self.file_info[index]) + "\n")

        position = np.array(position, dtype=np.int)

        return contig, contig_start, contig_end, chunk_id, image, position

    def __len__(self):
        return len(self.file_info)
