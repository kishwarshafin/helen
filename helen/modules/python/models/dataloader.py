from torch.utils.data import Dataset
import torchvision.transforms as transforms
import h5py
import sys
from helen.modules.python.TextColor import TextColor
from helen.modules.python.FileManager import FileManager


class SequenceDataset(Dataset):
    """
    This class implements the dataset class for the dataloader to use.
    This version is intended to use with train.py and test.py as this method loads labels with the images.
    It initializes all the given images and returns each image through __getitem__.
    """
    def __init__(self, image_directory):
        """
        This method initializes the dataset by loading all the image information. It creates a sequential list
        call all_images from which we can grab images iteratively through __getitem__.
        :param image_directory: Path to a directory where all the images are saved.
        """
        # transformer to convert loaded objects to tensors
        self.transform = transforms.Compose([transforms.ToTensor()])

        # a list of file-image pairs, where we have (file_name, image_name) as values so we can fetch images
        # from the list of files.
        file_image_pair = []

        # get all the h5 files that we have in the directory
        hdf_files = FileManager.get_file_paths_from_directory(image_directory)
        for hdf5_file_path in hdf_files:
            # for each of the files get all the images
            with h5py.File(hdf5_file_path, 'r') as hdf5_file:
                # check if marginpolish somehow generated an empty file
                if 'images' in hdf5_file:
                    image_names = list(hdf5_file['images'].keys())

                    # save the file-image pair to the list
                    for image_name in image_names:
                        file_image_pair.append((hdf5_file_path, image_name))
                else:
                    sys.stderr.write(TextColor.YELLOW + "WARN: NO IMAGES FOUND IN FILE: "
                                     + hdf5_file_path + "\n" + TextColor.END)

        # save the list to all_images so we can access the list inside other methods
        self.all_images = file_image_pair

    def __getitem__(self, index):
        """
        This method returns a single object. Dataloader uses this method to load images and then minibatches the loaded
        images
        :param index: Index indicating which image from all_images to be loaded
        :return:
        """
        # get the file path and the name of the image
        hdf5_filepath, image_name = self.all_images[index]

        # load the image and the label
        with h5py.File(hdf5_filepath, 'r') as hdf5_file:
            image = hdf5_file['images'][image_name]['image'][()]
            label_base = hdf5_file['images'][image_name]['label_base'][()]
            label_run_length = hdf5_file['images'][image_name]['label_run_length'][()]

        return image, label_base, label_run_length

    def __len__(self):
        """
        Returns the length of the dataset
        :return:
        """
        return len(self.all_images)
