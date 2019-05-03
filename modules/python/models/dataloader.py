from torch.utils.data import Dataset
import torchvision.transforms as transforms
import h5py
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
        A HDF5 file path
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
        # load the image
        hdf5_filepath, image_name = self.all_images[index]

        with h5py.File(hdf5_filepath, 'r') as hdf5_file:
            image = hdf5_file['images'][image_name]['image'][()]
            label_base = hdf5_file['images'][image_name]['label_base'][()]
            label_run_length = hdf5_file['images'][image_name]['label_run_length'][()]

        return image, label_base, label_run_length

    def __len__(self):
        return len(self.all_images)
