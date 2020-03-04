import sys
import torch

# Custom generator for our dataset
from helen.modules.python.models.train import train
from helen.modules.python.models.train_distributed import train_distributed
from helen.modules.python.Options import TrainOptions
from helen.modules.python.FileManager import FileManager
from helen.modules.python.TextColor import TextColor

"""
The train module of HELEN trains a deep neural network to perform a multi-task classification. It takes a set of
labeled images from MarginPolish and trains the model to predict a base and the run-length of the base using a 
gated recurrent unit (GRU) based model. This script is the interface to the training module. 
"""


class TrainModule:
    """
    Train module that provides an interface to the train method of HELEN.
    """
    def __init__(self, train_file, test_file, gpu_mode, device_ids, max_epochs, batch_size, num_workers,
                 retrain_model, retrain_model_path, model_dir, stats_dir):
        self.train_file = train_file
        self.test_file = test_file
        self.gpu_mode = gpu_mode
        self.device_ids = device_ids
        self.model_dir = model_dir
        self.epochs = max_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.retrain_model = retrain_model
        self.retrain_model_path = retrain_model_path
        self.stats_dir = stats_dir
        self.hidden_size = TrainOptions.HIDDEN_SIZE
        self.gru_layers = TrainOptions.GRU_LAYERS
        self.learning_rate = 0.0001
        self.weight_decay = 0

    def train_model(self):
        # train a model
        train(self.train_file,
              self.test_file,
              self.batch_size,
              self.epochs,
              self.gpu_mode,
              self.num_workers,
              self.retrain_model,
              self.retrain_model_path,
              self.gru_layers,
              self.hidden_size,
              self.learning_rate,
              self.weight_decay,
              self.model_dir,
              self.stats_dir,
              not_hyperband=True)

    def train_model_gpu(self):
        """
        DO DISTRIBUTED GPU INFERENCE. THIS MODE WILL ENABLE ONE MODEL PER GPU
        """
        if not torch.cuda.is_available():
            sys.stderr.write(TextColor.RED + "ERROR: TORCH IS NOT BUILT WITH CUDA.\n" + TextColor.END)
            sys.stderr.write(TextColor.RED + "SEE TORCH CAPABILITY:\n$ python3\n"
                                             ">>> import torch \n"
                                             ">>> torch.cuda.is_available()\n If true then cuda is avilable"
                             + TextColor.END)
            exit(1)

        # Now see which devices to use
        if self.device_ids is None:
            total_gpu_devices = torch.cuda.device_count()
            sys.stderr.write(TextColor.GREEN + "INFO: TOTAL GPU AVAILABLE: " + str(total_gpu_devices) + "\n" + TextColor.END)
            device_ids = [i for i in range(0, total_gpu_devices)]
            callers = total_gpu_devices
        else:
            device_ids = [int(i) for i in self.device_ids.split(',')]
            for device_id in device_ids:
                major_capable, minor_capable = torch.cuda.get_device_capability(device=device_id)
                if major_capable < 0:
                    sys.stderr.write(TextColor.RED + "ERROR: GPU DEVICE: " + str(device_id) + " IS NOT CUDA CAPABLE.\n" + TextColor.END)
                    sys.stderr.write(TextColor.GREEN + "Try running: $ python3\n"
                                                       ">>> import torch \n"
                                                       ">>> torch.cuda.get_device_capability(device="
                                     + str(device_id) + ")\n" + TextColor.END)
                else:
                    sys.stderr.write(TextColor.GREEN + "INFO: CAPABILITY OF GPU#" + str(device_id)
                                     + ":\t" + str(major_capable) + "-" + str(minor_capable) + "\n" + TextColor.END)
            callers = len(device_ids)

        if callers == 0:
            sys.stderr.write(TextColor.RED + "ERROR: NO GPU AVAILABLE BUT GPU MODE IS SET\n" + TextColor.END)
            exit()

        # train a model
        train_distributed(self.train_file,
                          self.test_file,
                          self.batch_size,
                          self.epochs,
                          self.gpu_mode,
                          self.num_workers,
                          self.retrain_model,
                          self.retrain_model_path,
                          self.gru_layers,
                          self.hidden_size,
                          self.learning_rate,
                          self.weight_decay,
                          self.model_dir,
                          self.stats_dir,
                          device_ids,
                          callers,
                          train_mode=True)


def train_interface(train_dir, test_dir, gpu_mode, device_ids, epoch_size, batch_size, num_workers, output_dir,
                    retrain_model, retrain_model_path):
    """
    Interface to perform training
    :param train_dir: Path to directory containing training images
    :param test_dir: Path to directory containing training images
    :param gpu_mode: GPU mode
    :param device_ids: Device IDs of devices to use for GPU inference
    :param epoch_size: Number of epochs to train on
    :param batch_size: Batch size
    :param num_workers: Number of workers for data loading
    :param output_dir: Path to directory to save model
    :param retrain_model: If you want to retrain an existing model
    :param retrain_model_path: Path to the model you want to retrain
    :return:
    """
    model_out_dir, stats_dir = FileManager.handle_train_output_directory(output_dir)
    tm = TrainModule(train_dir,
                     test_dir,
                     gpu_mode,
                     device_ids,
                     epoch_size,
                     batch_size,
                     num_workers,
                     retrain_model,
                     retrain_model_path,
                     model_out_dir,
                     stats_dir)

    if gpu_mode:
        tm.train_model_gpu()
    else:
        tm.train_model()
