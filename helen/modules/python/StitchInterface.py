import h5py
import sys
import os
from os.path import isfile, join
from os import listdir
from helen.modules.python.Stitch import Stitch
from helen.modules.python.TextColor import TextColor
from helen.modules.python.FileManager import FileManager
"""
The stitch module generates a consensus sequence from all the predictions we generated from call_consensus.py.

The method is as follows:
    1) MarginPolish
     - MarginPolish takes a "Region" which is usually 1000 bases and generates pileup summaries for them.
     - The region often is more than 1000 bases but for mini-batch to work, we make sure all the images are 1000 bases
       so MarginPolish chunks these Regions into images providing each Regional image with a chunk id.
    2) HELEN call consensus
     - Call Consensus generates a prediction for each of the images, but while saving it makes sure all the images
       that belong to the same region, gets saved under the same chunk_prefix, making sure all the sequences can
     - be aggregated easily.
    3) HELEN stitch
     -  The stitch method that loads one contig at a time and grabs all the "regions" with predictions. The regional
        chunks are easily stitched without having to do an overlap as the positions inside a region is consistent.
     - For regions that are adjacent, HELEN uses a local smith-waterman alignment to find an anchor point to
       stitch sequences from two adjacent sequences together.
"""


def get_file_paths_from_directory(directory_path):
    """
    Returns all paths of files given a directory path
    :param directory_path: Path to the directory
    :return: A list of paths of files
    """
    file_paths = [os.path.abspath(join(directory_path, file)) for file in listdir(directory_path)
                  if isfile(join(directory_path, file)) and file[-3:] == 'hdf']
    return file_paths


def perform_stitch(input_directory, output_path, output_prefix, threads):
    """
    This method gathers all contigs and calls the stitch module for each contig.
    :param input_directory: Path to the directory containing input files.
    :param output_path: Path to the output_consensus_sequence
    :param output_prefix: Output file's prefix
    :param threads: Number of threads to use
    :return:
    """
    # get all the files
    all_prediction_files = get_file_paths_from_directory(input_directory)

    # we gather all the contigs
    all_contigs = set()

    # get contigs from all of the files
    for prediction_file in sorted(all_prediction_files):
        with h5py.File(prediction_file, 'r') as hdf5_file:
            if 'predictions' in hdf5_file:
                contigs = list(hdf5_file['predictions'].keys())
                all_contigs.update(contigs)
            else:
                raise ValueError(TextColor.RED + "ERROR: INVALID HDF5 FILE, FILE DOES NOT CONTAIN predictions KEY.\n"
                                 + TextColor.END)
    # convert set to a list
    all_contigs = list(all_contigs)

    # get output directory
    output_dir = FileManager.handle_output_directory(output_path)

    # open an output fasta file
    # we should really use a fasta handler for this, I don't like this.
    output_filename = os.path.join(output_dir, output_prefix + '.fa')
    consensus_fasta_file = open(output_filename, 'w')
    sys.stderr.write(TextColor.GREEN + "INFO: OUTPUT FILE: " + output_filename + "\n" + TextColor.END)

    # for each contig
    for i, contig in enumerate(sorted(all_contigs)):
        log_prefix = "{:04d}".format(i) + "/" + "{:04d}".format(len(contigs)) + ":"
        sys.stderr.write(TextColor.GREEN + "INFO: " + str(log_prefix) + " PROCESSING CONTIG: " + contig + "\n"
                         + TextColor.END)

        # get all the chunk keys
        chunk_name_tuple = list()
        for prediction_file in all_prediction_files:
            with h5py.File(prediction_file, 'r') as hdf5_file:
                # check if the contig is contained in this file
                if contig not in list(hdf5_file['predictions'].keys()):
                    continue

                # if contained then get the chunks
                chunk_keys = sorted(hdf5_file['predictions'][contig].keys())
                for chunk_key in chunk_keys:
                    chunk_contig_start = hdf5_file['predictions'][contig][chunk_key]['contig_start'][()]
                    chunk_contig_end = hdf5_file['predictions'][contig][chunk_key]['contig_end'][()]
                    chunk_name_tuple.append((prediction_file, chunk_key, chunk_contig_start, chunk_contig_end))

        # call stitch to generate a sequence for this contig
        stich_object = Stitch()
        consensus_sequence = stich_object.create_consensus_sequence(contig, chunk_name_tuple, threads)
        sys.stderr.write(TextColor.BLUE + "INFO: " + str(log_prefix) + " FINISHED PROCESSING " + contig
                         + ", POLISHED SEQUENCE LENGTH: " + str(len(consensus_sequence)) + ".\n" + TextColor.END)

        # if theres a sequence then write it to the file
        if consensus_sequence is not None and len(consensus_sequence) > 0:
            consensus_fasta_file.write('>' + contig + "\n")
            consensus_fasta_file.write(consensus_sequence+"\n")