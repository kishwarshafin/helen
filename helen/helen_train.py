import argparse
import sys
import torch
from helen.version import __version__
from helen.modules.python.TextColor import TextColor
from helen.modules.python.TrainInterface import train_interface
from helen.modules.python.TestInterface import test_interface


def add_train_arguments(parser):
    """
    Add arguments to a parser for sub-command "train"
    :param parser: argeparse object
    :return:
    """
    parser.add_argument(
        "--train_image_dir",
        type=str,
        required=True,
        help="Path to directory containing labeled images for training."
    )
    parser.add_argument(
        "--test_image_dir",
        type=str,
        required=True,
        help="Path to directory containing labeled images for testing the models."
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
        "--output_dir",
        type=str,
        required=False,
        default='./model',
        help="Output directory path for model to save."
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
        default=False,
        action='store_true',
        help="If set then PyTorch will use GPUs for inference. CUDA required."
    )
    parser.add_argument(
        "-d_ids",
        "--device_ids",
        type=str,
        required=False,
        default=None,
        help="List of gpu device ids to use for inference.\n"
             "Example usage: --device_ids 0,1,2 (this will create three callers in id 'cuda:0, cuda:1 and cuda:2'\n"
             "If none then it will use all available devices."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        required=False,
        default=16,
        help="Number of workers to assign to the dataloader."
    )
    return parser


def add_test_arguments(parser):
    """
    Add arguments to a parser for sub-command "test"
    :param parser: argeparse object
    :return:
    """
    parser.add_argument(
        "--test_image_dir",
        type=str,
        required=True,
        help="Path to directory containing labeled images for testing the models."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=100,
        help="Batch size for training, default is 100."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=False,
        default='./model',
        help="Path of the model to load and retrain"
    )
    parser.add_argument(
        "--gpu_mode",
        action='store_true',
        help="If true then cuda is on."
    )
    parser.add_argument(
        "--print_details",
        action='store_true',
        help="If set then print details on mismatch cases."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default='./debug_output',
        help="Output file name."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        required=False,
        default=40,
        help="Epoch size for training iteration."
    )
    return parser


def add_hyperband_arguments(parser):
    """
    Add arguments to a parser for sub-command "stitch"
    :param parser: argeparse object
    :return:
    """
    parser.add_argument(
        "--train_image_dir",
        type=str,
        required=True,
        help="Path to directory containing labeled images for training."
    )
    parser.add_argument(
        "--test_image_dir",
        type=str,
        required=True,
        help="Path to directory containing labeled images for testing the models."
    )
    parser.add_argument(
        "--gpu_mode",
        action='store_true',
        help="If true then cuda is on."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default='./debug_output',
        help="Output file name."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=100,
        help="Batch size for training, default is 100."
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        required=False,
        default=10,
        help="Epoch size for training iteration."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        required=False,
        default=40,
        help="Epoch size for training iteration."
    )
    return parser


def main():
    """
    Main interface for HELEN training models:
    1) Train
    2) Test
    3) Hyperband
    """
    parser = argparse.ArgumentParser(
        description="The train module of HELEN trains a deep neural network to perform a multi-task classification. "
                    "It takes a set of labeled images from MarginPolish and trains the model to predict a base and "
                    "the run-length of the base using a gated recurrent unit (GRU) based model. "
                    "This script is the interface to the training module.",
                    formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "--version",
        default=False,
        action='store_true',
        help="Show version."
    )

    subparsers = parser.add_subparsers(dest='sub_command')
    # subparsers.required = True

    parser_train = subparsers.add_parser('train', help="Train a HELEN model. Requires a set of labeled images.")
    add_train_arguments(parser_train)

    parser_test = subparsers.add_parser('test', help="Test a model. Requires a set of labeled images")
    add_test_arguments(parser_test)

    subparsers.add_parser('torch_stat', help="See PyTorch configuration.")
    subparsers.add_parser('version', help="Show program version.")

    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.sub_command == 'train':
        sys.stderr.write(TextColor.GREEN + "INFO: TRAIN MODULE SELECTED\n" + TextColor.END)
        train_interface(FLAGS.train_image_dir,
                        FLAGS.test_image_dir,
                        FLAGS.gpu_mode,
                        FLAGS.device_ids,
                        FLAGS.epoch_size,
                        FLAGS.batch_size,
                        FLAGS.num_workers,
                        FLAGS.output_dir,
                        FLAGS.retrain_model,
                        FLAGS.retrain_model_path)

    elif FLAGS.sub_command == 'test':
        sys.stderr.write(TextColor.GREEN + "INFO: TEST MODULE SELECTED\n" + TextColor.END)
        test_interface(FLAGS.test_image_dir,
                       FLAGS.batch_size,
                       FLAGS.gpu_mode,
                       FLAGS.num_workers,
                       FLAGS.model_path,
                       FLAGS.output_dir,
                       FLAGS.print_details)

    elif FLAGS.sub_command == 'torch_stat':
        sys.stderr.write(TextColor.YELLOW + "TORCH VERSION: " + TextColor.END + str(torch.__version__) + "\n\n")
        sys.stderr.write(TextColor.YELLOW + "PARALLEL CONFIG:\n" + TextColor.END)
        print(torch.__config__.parallel_info())
        sys.stderr.write(TextColor.YELLOW + "BUILD CONFIG:\n" + TextColor.END)
        print(*torch.__config__.show().split("\n"), sep="\n")

        sys.stderr.write(TextColor.GREEN + "CUDA AVAILABLE: " + TextColor.END + str(torch.cuda.is_available()) + "\n")
        sys.stderr.write(TextColor.GREEN + "GPU DEVICES: " + TextColor.END + str(torch.cuda.device_count()) + "\n")

    elif FLAGS.version is True:
        print("HELEN VERSION: ", __version__)

    else:
        sys.stderr.write(TextColor.RED + "ERROR: NO SUBCOMMAND SELECTED. "
                                         "PLEASE SELECT ONE OF THE AVAIABLE SUB-COMMANDS.\n"
                         + TextColor.END)
        parser.print_help()


if __name__ == '__main__':
    main()