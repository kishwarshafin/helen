import argparse
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from modules.python.models.dataloader_predict import SequenceDataset
from torchvision import transforms
import multiprocessing
from modules.python.TextColor import TextColor
from collections import defaultdict
from modules.python.vcf_writer import VCFWriter
from modules.python.FileManager import FileManager
import operator
import pickle
from tqdm import tqdm
import os
import time
import numpy as np
from modules.python.models.ModelHander import ModelHandler
from modules.python.Options import ImageSizeOptions, TrainOptions
"""
This script uses a trained model to call variants on a given set of images generated from the genome.
The process is:
- Create a prediction table/dictionary using a trained neural network
- Convert those predictions to a VCF file

INPUT:
- A trained model
- Set of images for prediction

Output:
- A VCF file containing all the variants.
"""

prediction_dict = defaultdict(lambda: [0.0] * ImageSizeOptions.TOTAL_LABELS)
position_dict = defaultdict(set)
chromosome_list = set()
label_decoder = {1: 'A', 2: 'C', 3: 'G', 4: 'T', 0: ''}


def predict(test_file, model_path, batch_size, num_workers, gpu_mode):
    """
    Create a prediction table/dictionary of an images set using a trained model.
    :param test_file: File to predict on
    :param batch_size: Batch size used for prediction
    :param model_path: Path to a trained model
    :param gpu_mode: If true, predictions will be done over GPU
    :param num_workers: Number of workers to be used by the dataloader
    :return: Prediction dictionary
    """
    sys.stderr.write(TextColor.PURPLE + 'Loading data\n' + TextColor.END)

    # data loader
    test_data = SequenceDataset(test_file)
    test_loader = DataLoader(test_data,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)

    transducer_model, hidden_size, gru_layers, prev_ite = \
        ModelHandler.load_simple_model_for_training(model_path,
                                                    input_channels=ImageSizeOptions.IMAGE_CHANNELS,
                                                    image_features=ImageSizeOptions.IMAGE_HEIGHT,
                                                    seq_len=ImageSizeOptions.SEQ_LENGTH,
                                                    num_classes=ImageSizeOptions.TOTAL_LABELS)
    transducer_model.eval()

    if gpu_mode:
        transducer_model = torch.nn.DataParallel(transducer_model).cuda()
    sys.stderr.write(TextColor.CYAN + 'MODEL LOADED\n')

    with torch.no_grad():
        for images, chromosome, position, index in tqdm(test_loader, ncols=50):
            if gpu_mode:
                # encoder_hidden = encoder_hidden.cuda()
                images = images.cuda()

            hidden = torch.zeros(images.size(0), 2 * TrainOptions.GRU_LAYERS, TrainOptions.HIDDEN_SIZE)

            if gpu_mode:
                hidden = hidden.cuda()

            for i in range(0, ImageSizeOptions.SEQ_LENGTH, TrainOptions.WINDOW_JUMP):
                if i + TrainOptions.TRAIN_WINDOW > ImageSizeOptions.SEQ_LENGTH:
                    break
                chunk_start = i
                chunk_end = i + TrainOptions.TRAIN_WINDOW
                # chunk all the data
                image_chunk = images[:, chunk_start:chunk_end]
                position_chunk = position[:, chunk_start:chunk_end]
                index_chunk = index[:, chunk_start:chunk_end]

                # run inference
                output_, hidden = transducer_model(image_chunk, hidden)

                # do softmax and get prediction
                m = nn.Softmax(dim=2)
                soft_probs = m(output_)
                output_preds = soft_probs.cpu()
                max_value, predicted_label = torch.max(output_preds, dim=2)

                # convert everything to list
                max_value = max_value.numpy().tolist()
                predicted_label = predicted_label.numpy().tolist()
                position_chunk = position_chunk.numpy().tolist()
                index_chunk = index_chunk.numpy().tolist()

                assert(len(index_chunk) == len(position_chunk) == len(max_value) == len(predicted_label))

                for ii in range(0, len(position_chunk)):
                    counter = 0

                    for pos, idx, p, label in zip(position_chunk[ii],
                                                  index_chunk[ii],
                                                  max_value[ii],
                                                  predicted_label[ii]):
                        if pos < 0 or idx < 0:
                            continue
                        prediction_dict[(chromosome[ii], pos, idx)][label] += p
                        counter += 1
                        position_dict[chromosome[ii]].add((chromosome[ii], pos, idx))
                        chromosome_list.add(chromosome[ii])


def polish_genome(csv_file, model_path, batch_size, num_workers, output_dir, gpu_mode, max_threads):
    sys.stderr.write(TextColor.GREEN + "INFO: " + TextColor.END + "OUTPUT DIRECTORY: " + output_dir + "\n")
    predict(csv_file, model_path, batch_size, num_workers, gpu_mode)
    sys.stderr.write(TextColor.GREEN + "INFO: " + TextColor.END + "PREDICTION GENERATED SUCCESSFULLY.\n")
    sys.stderr.write(TextColor.GREEN + "INFO: " + TextColor.END + "COMPILING PREDICTIONS TO CALL VARIANTS.\n")

    fasta_file = open(output_dir + "helen_polished.fa", 'w')

    for chromosome in chromosome_list:
        pos_list = list(position_dict[chromosome])
        pos_list = sorted(list(pos_list), key=lambda element: (element[0], element[1], element[2]))

        chr_prev, pos_prev, indx_prev = pos_list[0]
        for i in range(1, len(pos_list)):
            chr, pos, indx = pos_list[i]
            if indx > 0:
                continue
            else:
                if pos - pos_prev != 1:
                    print(pos_prev, pos)
                    exit()
                chr_prev, pos_prev, indx_prev = pos_list[i]
        dict_fetch = operator.itemgetter(*pos_list)
        predicted_labels = list(dict_fetch(prediction_dict))
        predicted_labels = np.argmax(np.array(predicted_labels), axis=1).tolist()
        sequence = ''.join([label_decoder[x] for x in predicted_labels])

        if len(sequence) > 0:
            fasta_file.write('>'+chromosome+"\n")
            fasta_file.write(sequence+"\n")


def handle_output_directory(output_dir):
    """
    Process the output directory and return a valid directory where we save the output
    :param output_dir: Output directory path
    :return:
    """
    # process the output directory
    if output_dir[-1] != "/":
        output_dir += "/"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    return output_dir


if __name__ == '__main__':
    '''
    Processes arguments and performs tasks.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_file",
        type=str,
        required=True,
        help="HDF5 file containing all image segments for prediction."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=100,
        help="Batch size for testing, default is 100."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        required=False,
        default=4,
        help="Batch size for testing, default is 100."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default='vcf_output',
        help="Output directory."
    )
    parser.add_argument(
        "--max_threads",
        type=int,
        default=8,
        help="Number of maximum threads for this region."
    )
    parser.add_argument(
        "--gpu_mode",
        type=bool,
        default=False,
        help="If true then cuda is on."
    )
    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.output_dir = handle_output_directory(FLAGS.output_dir)
    polish_genome(FLAGS.image_file,
                  FLAGS.model_path,
                  FLAGS.batch_size,
                  FLAGS.num_workers,
                  FLAGS.output_dir,
                  FLAGS.gpu_mode,
                  FLAGS.max_threads)

