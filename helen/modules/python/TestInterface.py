import os
import sys
import matplotlib.pyplot as plt
import numpy as np

from helen.modules.python.models.test_debug import test
from helen.modules.python.models.ModelHander import ModelHandler
from helen.modules.python.TextColor import TextColor
from helen.modules.python.FileManager import FileManager
from helen.modules.python.Options import ImageSizeOptions
"""
FREEZE THIS BRANCH TO HAVE 1 WINDOW!!
Train a model and save the model that performs best.

Input:
- A train CSV containing training image set information (usually chr1-18)
- A test CSV containing testing image set information (usually chr19)

Output:
- A trained model
"""


def save_rle_confusion_matrix(stats_dictionary, output_directory):
    # plot confusion matrix
    fig, ax = plt.subplots(figsize=(20, 20))
    cf = np.array(stats_dictionary['rle_confusion_matrix'], dtype=np.int)
    im = ax.imshow(cf)
    rle_labels = [str(i) for i in range(0, ImageSizeOptions.TOTAL_RLE_LABELS)]

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(rle_labels)))
    ax.set_yticks(np.arange(len(rle_labels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(rle_labels)
    ax.set_yticklabels(rle_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(rle_labels)):
        for j in range(len(rle_labels)):
            if cf[i, j] > 0:
                if i == j:
                    text = ax.text(j, i, cf[i, j], ha="center", va="center", color="g")
                else:
                    text = ax.text(j, i, cf[i, j], ha="center", va="center", color="r")

    ax.set_title("RLE Confusion Matrix")
    fig.tight_layout()
    # plt.show()
    plt.savefig(output_directory + "/RLE_CONFUSION_MATRIX.png", dpi=100)


def save_base_confusion_matrix(stats_dictionary, output_directory):
    # plot confusion matrix
    fig, ax = plt.subplots(figsize=(5, 5))
    cf = np.array(stats_dictionary['base_confusion_matrix'], dtype=np.int)
    im = ax.imshow(cf)
    base_labels = [str(i) for i in ['-', 'A', 'C', 'T', 'G']]

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(base_labels)))
    ax.set_yticks(np.arange(len(base_labels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(base_labels)
    ax.set_yticklabels(base_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(base_labels)):
        for j in range(len(base_labels)):
            if cf[i, j] > 0:
                if i == j:
                    text = ax.text(j, i, cf[i, j], ha="center", va="center", color="g")
                else:
                    text = ax.text(j, i, cf[i, j], ha="center", va="center", color="r")

    ax.set_title("BASE Confusion Matrix")
    fig.tight_layout()
    # plt.show()
    plt.savefig(output_directory + "/BASE_CONFUSION_MATRIX.png", dpi=100)


def test_interface(test_file, batch_size, gpu_mode, num_workers, model_path, output_directory, print_details):
    """
    Test a trained model
    :param test_file: Path to directory containing test images
    :param batch_size: Batch size for training
    :param gpu_mode: If true the model will be trained on GPU
    :param num_workers: Number of workers for data loading
    :param model_path: Path to a saved model
    :param output_directory: Path to output_directory
    :param print_details: If true then logs will be printed
    :return:
    """
    sys.stderr.write(TextColor.PURPLE + 'Loading data\n' + TextColor.END)

    output_directory = FileManager.handle_output_directory(output_directory)

    if os.path.isfile(model_path) is False:
        sys.stderr.write(TextColor.RED + "ERROR: INVALID PATH TO MODEL\n")
        exit(1)

    sys.stderr.write(TextColor.GREEN + "INFO: MODEL LOADING\n" + TextColor.END)

    transducer_model, hidden_size, gru_layers, prev_ite = \
        ModelHandler.load_simple_model(model_path,
                                       input_channels=ImageSizeOptions.IMAGE_CHANNELS,
                                       image_features=ImageSizeOptions.IMAGE_HEIGHT,
                                       seq_len=ImageSizeOptions.SEQ_LENGTH,
                                       num_base_classes=ImageSizeOptions.TOTAL_BASE_LABELS,
                                       num_rle_classes=ImageSizeOptions.TOTAL_RLE_LABELS)

    sys.stderr.write(TextColor.GREEN + "INFO: MODEL LOADED\n" + TextColor.END)

    if print_details and gpu_mode:
        sys.stderr.write(TextColor.GREEN + "INFO: GPU MODE NOT AVAILABLE WHEN PRINTING DETAILS. "
                                           "SETTING GPU MODE TO FALSE.\n" + TextColor.END)
        gpu_mode = False

    if gpu_mode:
        transducer_model = transducer_model.cuda()

    stats_dictionary = test(test_file, batch_size, gpu_mode, transducer_model, num_workers,
                            gru_layers, hidden_size, num_base_classes=ImageSizeOptions.TOTAL_BASE_LABELS,
                            num_rle_classes=ImageSizeOptions.TOTAL_RLE_LABELS,
                            output_directory=output_directory,
                            print_details=print_details)

    save_rle_confusion_matrix(stats_dictionary, output_directory)
    save_base_confusion_matrix(stats_dictionary, output_directory)

    sys.stderr.write(TextColor.PURPLE + 'DONE\n' + TextColor.END)