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
from modules.python.DataStore_predict import DataStore
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


def predict(test_file, output_filename, model_path, batch_size, num_workers, gpu_mode):
    """
    Create a prediction table/dictionary of an images set using a trained model.
    :param test_file: File to predict on
    :param batch_size: Batch size used for prediction
    :param model_path: Path to a trained model
    :param gpu_mode: If true, predictions will be done over GPU
    :param num_workers: Number of workers to be used by the dataloader
    :return: Prediction dictionary
    """
    prediction_data_file = DataStore(output_filename, mode='w')
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
                                                    num_base_classes=ImageSizeOptions.TOTAL_BASE_LABELS,
                                                    num_rle_classes=ImageSizeOptions.TOTAL_RLE_LABELS)
    transducer_model.eval()

    if gpu_mode:
        transducer_model = torch.nn.DataParallel(transducer_model).cuda()
    sys.stderr.write(TextColor.CYAN + 'MODEL LOADED\n')

    with torch.no_grad():
        for contig, contig_start, contig_end, chunk_id, images, position in tqdm(test_loader, ncols=50):
            if gpu_mode:
                # encoder_hidden = encoder_hidden.cuda()
                images = images.cuda()

            hidden = torch.zeros(images.size(0), 2 * TrainOptions.GRU_LAYERS, TrainOptions.HIDDEN_SIZE)

            if gpu_mode:
                hidden = hidden.cuda()

            prediction_base_dict = np.zeros((images.size(0), images.size(1), ImageSizeOptions.TOTAL_BASE_LABELS))
            prediction_rle_dict = np.zeros((images.size(0), images.size(1), ImageSizeOptions.TOTAL_RLE_LABELS))

            for i in range(0, ImageSizeOptions.SEQ_LENGTH, TrainOptions.WINDOW_JUMP):
                if i + TrainOptions.TRAIN_WINDOW > ImageSizeOptions.SEQ_LENGTH:
                    break
                chunk_start = i
                chunk_end = i + TrainOptions.TRAIN_WINDOW
                # chunk all the data
                image_chunk = images[:, chunk_start:chunk_end]

                # run inference
                output_base, output_rle, hidden = transducer_model(image_chunk, hidden)

                # do softmax and get prediction
                m = nn.Softmax(dim=2)
                soft_probs = m(output_base)
                output_preds = soft_probs.cpu()
                base_max_value, predicted_base_label = torch.max(output_preds, dim=2)

                # convert everything to list
                base_max_value = base_max_value.numpy().tolist()
                predicted_base_label = predicted_base_label.numpy().tolist()

                # do softmax and get prediction for rle
                m_rle = nn.Softmax(dim=2)
                rle_soft_probs = m_rle(output_rle)
                rle_output_preds = rle_soft_probs.cpu()
                rle_max_value, predicted_rle_labels = torch.max(rle_output_preds, dim=2)

                # convert everything to list
                rle_max_value = rle_max_value.numpy().tolist()
                predicted_rle_labels = predicted_rle_labels.numpy().tolist()

                assert(len(base_max_value) == len(predicted_base_label) == len(predicted_rle_labels))

                for ii in range(0, len(predicted_base_label)):
                    chunk_pos = chunk_start
                    for p_base, p_rle, base, rle in zip(base_max_value[ii],
                                                        rle_max_value[ii],
                                                        predicted_base_label[ii],
                                                        predicted_rle_labels[ii]):
                        prediction_base_dict[ii][chunk_pos][base] += p_base
                        prediction_rle_dict[ii][chunk_pos][rle] += p_rle
                        chunk_pos += 1
            predicted_base_labels = np.argmax(np.array(prediction_base_dict), axis=2)
            predicted_rle_labels = np.argmax(np.array(prediction_rle_dict), axis=2)

            for i in range(images.size(0)):
                prediction_data_file.write_prediction(contig[i], contig_start[i], contig_end[i], chunk_id[i],
                                                      position[i], predicted_base_labels[i], predicted_rle_labels[i])


def polish_genome(csv_file, model_path, batch_size, num_workers, output_dir, gpu_mode):
    sys.stderr.write(TextColor.GREEN + "INFO: " + TextColor.END + "OUTPUT DIRECTORY: " + output_dir + "\n")
    output_filename = output_dir + "helen_predictions.hdf"
    predict(csv_file, output_filename, model_path, batch_size, num_workers, gpu_mode)
    sys.stderr.write(TextColor.GREEN + "INFO: " + TextColor.END + "PREDICTION GENERATED SUCCESSFULLY.\n")
    sys.stderr.write(TextColor.GREEN + "INFO: " + TextColor.END + "COMPILING PREDICTIONS TO CALL VARIANTS.\n")


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
                  FLAGS.gpu_mode)

