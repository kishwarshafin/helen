import argparse
import sys
import torch
from helen.version import __version__
from helen.modules.python.TextColor import TextColor
from helen.modules.python.PolishInterface import polish_genome
from helen.modules.python.CallConsensusInterface import call_consensus
from helen.modules.python.StitchInterface import perform_stitch
from helen.modules.python.DownloadModel import download_models


def add_polish_arguments(parser):
    """
    Add arguments to a parser for sub-command "polish"
    :param parser: argeparse object
    :return:
    """
    parser.add_argument(
        "-i",
        "--image_dir",
        type=str,
        required=True,
        help="[REQUIRED] Path to a directory where all MarginPolish generated images are."
    )
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        required=True,
        help="[REQUIRED] Path to a trained model (pkl file). Please see our github page to see options."
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        required=False,
        default=512,
        help="Batch size for testing, default is 512. Please set to 512 or 1024 for a balanced execution time."
    )
    parser.add_argument(
        "-w",
        "--num_workers",
        type=int,
        required=False,
        default=8,
        help="Number of workers to assign to the dataloader."
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        required=False,
        default=1,
        help="Number of PyTorch threads to use, default is 1. This may be helpful during CPU-only inference."
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=False,
        default='./output/',
        help="Path to the output directory."
    )
    parser.add_argument(
        "-p",
        "--output_prefix",
        type=str,
        required=False,
        default="HELEN_prediction",
        help="Prefix for the output file. Default is: HELEN_prediction"
    )
    parser.add_argument(
        "-g",
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
        help="List of gpu device ids to use for inference. Only used in distributed setting.\n"
             "Example usage: --device_ids 0,1,2 (this will create three callers in id 'cuda:0, cuda:1 and cuda:2'\n"
             "If none then it will use all available devices."
    )
    parser.add_argument(
        "-c",
        "--callers",
        type=int,
        required=False,
        default=8,
        help="Total number of callers to spawn if doing CPU inference in distributed mode."
    )
    return parser


def add_call_consensus_arguments(parser):
    """
    Add arguments to a parser for sub-command "call_consensus"
    :param parser: argeparse object
    :return:
    """
    parser.add_argument(
        "-i",
        "--image_dir",
        type=str,
        required=True,
        help="[REQUIRED] Path to a directory where all MarginPolish generated images are."
    )
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        required=True,
        help="[REQUIRED] Path to a trained model (pkl file). Please see our github page to see options."
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        required=False,
        default=512,
        help="Batch size for testing, default is 512. Please set to 512 or 1024 for a balanced execution time."
    )
    parser.add_argument(
        "-w",
        "--num_workers",
        type=int,
        required=False,
        default=8,
        help="Number of workers to assign to the dataloader. Shouldg be 0 if using Docker."
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=False,
        default='./output/',
        help="Path to the output directory."
    )
    parser.add_argument(
        "-p",
        "--output_prefix",
        type=str,
        required=False,
        default="HELEN_prediction",
        help="Prefix for the output file. Default is: HELEN_prediction"
    )
    parser.add_argument(
        "-g",
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
        help="List of gpu device ids to use for inference. Only used in distributed setting.\n"
             "Example usage: --device_ids 0,1,2 (this will create three callers in id 'cuda:0, cuda:1 and cuda:2'\n"
             "If none then it will use all available devices."
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        required=False,
        default=16,
        help="Total available threads to use."
    )
    parser.add_argument(
        "-c",
        "--callers",
        type=int,
        required=False,
        default=8,
        help="Total number of callers to spawn if doing CPU inference in distributed mode."
    )
    return parser


def add_stitch_arguments(parser):
    """
    Add arguments to a parser for sub-command "stitch"
    :param parser: argeparse object
    :return:
    """
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        required=True,
        help="[REQUIRED] Path to a directory containing prediction files call consensus."
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="[REQUIRED] Path to the output directory."
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        required=True,
        help="[REQUIRED] Number of threads."
    )
    parser.add_argument(
        "-p",
        "--output_prefix",
        type=str,
        required=False,
        default="HELEN_consensus",
        help="Prefix for the output file. Default is: HELEN_consensus"
    )


def add_download_models_arguments(parser):
    """
    Add parameters for model download script
    :param parser: Parser object
    :return:
    """
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Path to directory where models will be saved."
    )
    return parser


def main():
    """
    Main interface for HELEN. The submodules supported as of now are these:
    1) Polish
    2) Call_consensus
    3) Stitch
    """
    parser = argparse.ArgumentParser(description="HELEN is a RNN based polisher for polishing ONT-based assemblies. "
                                                 "You can avail three commands with this script:\n"
                                                 "1) polish: Call consensus and stitch the predictions.\n"
                                                 "2) call_consensus: This module takes the summary images and a"
                                                 "trained neural network and generates predictions per base.\n"
                                                 "3) stitch: This module takes the inference files as input and "
                                                 "stitches them to generate a polished assembly.\n"
                                                 "4) download_model: Download available helen models\n"
                                                 "5) torch_stat: See the torch configuration\n"
                                                 "6) version: check HELEN version.\n",

                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--version",
        default=False,
        action='store_true',
        help="Show version."
    )

    subparsers = parser.add_subparsers(dest='sub_command')
    # subparsers.required = True

    parser_polish = subparsers.add_parser('polish', help="Run the polishing pipeline. This will run "
                                                         "call_consensus -> stitch one after another.\n"
                                                         "The outputs of each step can be run separately using\n"
                                                         "the appropriate sub-command.\n")
    add_polish_arguments(parser_polish)

    parser_call_consensus = subparsers.add_parser('call_consensus',
                                                  help="call_consensus.py script HELEN performs inference on a given "
                                                       "set of images generated by MarginPolish.\n"
                                                       "The input of this script is a directory containing all the\n"
                                                       "images generated by MarginPolish, a model and an\n"
                                                       "output directory.\n"
                                                       "OUTPUT: This module generates .hdf files which is used by the "
                                                       "stitch.py script to perform the final stitching.\n")
    add_call_consensus_arguments(parser_call_consensus)

    parser_stitch = subparsers.add_parser('stitch',
                                          help="stitch.py of HELEN generates a polished sequence from a given hdf"
                                               " file generated by call_consensus.py script. \nstitch.py is "
                                               " multi-threaded and faster run-time can be achieved on higher number"
                                               " of CPUs.\n"
                                               "OUTPUT: Is a FASTA file containing a polished consensus sequence.\n")
    add_stitch_arguments(parser_stitch)

    parser_download_model = subparsers.add_parser('download_models', help="Download available models.")
    add_download_models_arguments(parser_download_model)

    parser_torch_stat = subparsers.add_parser('torch_stat', help="See PyTorch configuration.")
    parser_torch_stat = subparsers.add_parser('version', help="Show program version.")

    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.sub_command == 'polish':
        sys.stderr.write(TextColor.GREEN + "INFO: POLISH MODULE SELECTED\n" + TextColor.END)
        polish_genome(FLAGS.image_dir,
                      FLAGS.model_path,
                      FLAGS.batch_size,
                      FLAGS.num_workers,
                      FLAGS.threads,
                      FLAGS.output_dir,
                      FLAGS.output_prefix,
                      FLAGS.gpu_mode,
                      FLAGS.device_ids,
                      FLAGS.callers)

    elif FLAGS.sub_command == 'call_consensus':
        sys.stderr.write(TextColor.GREEN + "INFO: CALL CONSENSUS MODULE SELECTED\n" + TextColor.END)
        call_consensus(FLAGS.image_dir,
                       FLAGS.model_path,
                       FLAGS.batch_size,
                       FLAGS.num_workers,
                       FLAGS.threads,
                       FLAGS.output_dir,
                       FLAGS.output_prefix,
                       FLAGS.gpu_mode,
                       FLAGS.device_ids,
                       FLAGS.callers)

    elif FLAGS.sub_command == 'stitch':
        sys.stderr.write(TextColor.GREEN + "INFO: STITCH MODULE SELECTED\n" + TextColor.END)
        perform_stitch(FLAGS.input_dir,
                       FLAGS.output_dir,
                       FLAGS.output_prefix,
                       FLAGS.threads)

    elif FLAGS.sub_command == 'download_models':
        sys.stderr.write(TextColor.GREEN + "INFO: DOWNLOAD MODELS SELECTED\n" + TextColor.END)
        download_models(FLAGS.output_dir)

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
