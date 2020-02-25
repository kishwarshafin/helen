import argparse
import sys
import torch

# Custom generator for our dataset
from modules.python.models.train import train
from modules.python.models.train_distributed import train_distributed
from modules.python.Options import TrainOptions
from modules.python.FileManager import FileManager
from modules.python.TextColor import TextColor

"""
The train module of HELEN trains a deep neural network to perform a multi-task classification. It takes a set of
labeled images from MarginPolish and trains the model to predict a base and the run-length of the base using a 
gated recurrent unit (GRU) based model. This script is the interface to the training module. 
"""


class TrainModule:
    """
    Train module that provides an interface to the train method of HELEN.
    """
    def __init__(self, train_file, test_file, gpu_mode, max_epochs, batch_size, num_workers,
                 retrain_model, retrain_model_path, model_dir, stats_dir):
        self.train_file = train_file
        self.test_file = test_file
        self.gpu_mode = gpu_mode
        self.log_directory = log_dir
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

    def train_model_distributed(self):
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

        total_gpu_devices = torch.cuda.device_count()

        total_gpu_devices = torch.cuda.device_count()
        sys.stderr.write(TextColor.GREEN + "INFO: TOTAL GPU AVAILABLE: " + str(total_gpu_devices) + "\n" + TextColor.END)
        device_ids = [i for i in range(0, total_gpu_devices)]
        total_callers = total_gpu_devices

        sys.stderr.write(TextColor.GREEN + "INFO: AVAILABLE GPU DEVICES: " + str(device_ids) + "\n" + TextColor.END)

        if total_callers == 0:
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
                          total_callers,
                          train_mode=True)


if __name__ == '__main__':
    '''
    Processes arguments and performs tasks.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="Training data description csv file."
    )
    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="Training data description csv file."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=100,
        help="Batch size for training, default is 100."
    )
    parser.add_argument(
        "--epoch_size",
        type=int,
        required=False,
        default=10,
        help="Epoch size for training iteration."
    )
    parser.add_argument(
        "--model_out",
        type=str,
        required=False,
        default='./model',
        help="Path and file_name to save model, default is ./model"
    )
    parser.add_argument(
        "--retrain_model",
        type=bool,
        default=False,
        help="If true then retrain a pre-trained mode."
    )
    parser.add_argument(
        "--retrain_model_path",
        type=str,
        default=False,
        help="Path to the model that will be retrained."
    )
    parser.add_argument(
        "--gpu_mode",
        type=bool,
        default=False,
        help="If true then cuda is on."
    )
    parser.add_argument(
        "--distributed",
        type=bool,
        default=False,
        help="If true then distributed training will be on."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        required=False,
        default=16,
        help="Number of workers to assign to the dataloader."
    )
    FLAGS, unparsed = parser.parse_known_args()
    model_out_dir, log_dir = FileManager.handle_train_output_directory(FLAGS.model_out)
    tm = TrainModule(FLAGS.train_file, FLAGS.test_file, FLAGS.gpu_mode, FLAGS.epoch_size, FLAGS.batch_size,
                     FLAGS.num_workers, FLAGS.retrain_model, FLAGS.retrain_model_path, model_out_dir, log_dir)
    if FLAGS.distributed:
        tm.train_model_distributed()
    else:
        tm.train_model()
