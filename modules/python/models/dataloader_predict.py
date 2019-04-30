import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import h5py
import torch
from modules.python.Options import ImageSizeOptions


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
        with h5py.File(self.file_info[index], 'r') as hdf5_file:
            contig = np.array2string(hdf5_file['contig'][()][0].astype(np.str)).replace("'", '')
            contig_start = hdf5_file['contig_start'][()][0].astype(np.int)
            contig_end = hdf5_file['contig_end'][()][0].astype(np.int)
            chunk_id = hdf5_file['feature_chunk_idx'][()][0].astype(np.int)
            image = hdf5_file['image'][()].astype(np.uint8)
            position = hdf5_file['position'][()].astype(np.int)

        if image.shape[0] < ImageSizeOptions.SEQ_LENGTH:
            total_empty_needed = ImageSizeOptions.SEQ_LENGTH - image.shape[0]
            empty_image_columns = np.array([[0] * ImageSizeOptions.IMAGE_HEIGHT] * total_empty_needed)
            image = np.append(image, empty_image_columns, 0)
            image = image.astype(np.uint8)

            empty_positions = np.array([[-1, -1, -1]] * total_empty_needed)
            position = np.append(position, empty_positions, 0)
            position = position.astype(np.int)

        if image.shape[0] < ImageSizeOptions.SEQ_LENGTH or position.shape[0] < ImageSizeOptions.SEQ_LENGTH:
            raise ValueError("IMAGE SIZE ERROR: " + str(self.file_info[index]) + " " + str(image.shape))

        return contig, contig_start, contig_end, chunk_id, image, position, self.file_info[index]

    def __len__(self):
        return len(self.file_info)
