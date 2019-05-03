import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import h5py
from modules.python.Options import ImageSizeOptions
from os.path import isfile, join
from os import listdir


def get_file_paths_from_directory(directory_path):
    """
    Returns all paths of files given a directory path
    :param directory_path: Path to the directory
    :return: A list of paths of files
    """
    file_paths = [join(directory_path, file) for file in listdir(directory_path) if isfile(join(directory_path, file))
                  and file[-2:] == 'h5']
    return file_paths


class SequenceDataset(Dataset):
    """
    Arguments:
        A CSV file path
    """

    def __init__(self, image_directory):
        self.transform = transforms.Compose([transforms.ToTensor()])
        file_image_pair = []

        hdf_files = get_file_paths_from_directory(image_directory)

        for hdf5_file_path in hdf_files:
            with h5py.File(hdf5_file_path, 'r') as hdf5_file:
                image_names = list(hdf5_file['images'].keys())

            for image_name in image_names:
                file_image_pair.append((hdf5_file_path, image_name))

        self.all_images = file_image_pair

    def __getitem__(self, index):

        hdf5_filepath, image_name = self.all_images[index]

        with h5py.File(hdf5_filepath, 'r') as hdf5_file:
            contig = np.array2string(hdf5_file['images'][image_name]['contig'][()][0].astype(np.str)).replace("'", '')
            contig_start = hdf5_file['images'][image_name]['contig_start'][()][0].astype(np.int)
            contig_end = hdf5_file['images'][image_name]['contig_end'][()][0].astype(np.int)
            chunk_id = hdf5_file['images'][image_name]['feature_chunk_idx'][()][0].astype(np.int)
            image = hdf5_file['images'][image_name]['image'][()].astype(np.uint8)
            position = hdf5_file['images'][image_name]['position'][()].astype(np.int)

        if image.shape[0] < ImageSizeOptions.SEQ_LENGTH:
            total_empty_needed = ImageSizeOptions.SEQ_LENGTH - image.shape[0]
            empty_image_columns = np.array([[0] * ImageSizeOptions.IMAGE_HEIGHT] * total_empty_needed)
            image = np.append(image, empty_image_columns, 0)
            image = image.astype(np.uint8)

            empty_positions = np.array([[-1, -1, -1]] * total_empty_needed)
            position = np.append(position, empty_positions, 0)
            position = position.astype(np.int)

        if image.shape[0] < ImageSizeOptions.SEQ_LENGTH or position.shape[0] < ImageSizeOptions.SEQ_LENGTH:
            raise ValueError("IMAGE SIZE ERROR: " + str(hdf5_filepath) + " " + str(image.shape))

        return contig, contig_start, contig_end, chunk_id, image, position, hdf5_filepath

    def __len__(self):
        return len(self.all_images)
