import h5py
import argparse
import sys
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
from os.path import isfile, join
from os import listdir
import concurrent.futures
import numpy as np
from collections import  defaultdict
import operator

BASE_ERROR_RATE = 0.2


def label_to_sequence(label):
    if label == 0:
        return ''
    if label <= 20:
        return 'A' * label
    if label <= 40:
        return 'C' * (label - 20)
    if label <= 60:
        return 'G' * (label - 40)
    if label <= 80:
        return 'T' * (label - 60)
    else:
        print("INVALID LABEL VALUE: ", label)


def get_file_paths_from_directory(directory_path):
    """
    Returns all paths of files given a directory path
    :param directory_path: Path to the directory
    :return: A list of paths of files
    """
    file_paths = [join(directory_path, file) for file in listdir(directory_path) if isfile(join(directory_path, file)) and file[-2:] == 'h5']
    return file_paths


def chunks(file_names, threads):
    """Yield successive n-sized chunks from l."""
    chunks = []
    for i in range(0, len(file_names), threads):
        chunks.append(file_names[i:i + threads])
    return chunks


def small_chunk_stitch(file_name, contig, small_chunk_keys):
    # for chunk_key in small_chunk_keys:
    hdf5_file = h5py.File(file_name, 'r')
    name_sequence_tuples = list()

    for chunk_name in small_chunk_keys:
        smaller_chunks = list(hdf5_file['predictions'][contig][chunk_name].keys())

        positions = set()
        prediction_dict = defaultdict()
        for chunk in smaller_chunks:
            predictions = hdf5_file['predictions'][contig][chunk_name][chunk]['predictions']
            index = hdf5_file['predictions'][contig][chunk_name][chunk]['index']
            position = hdf5_file['predictions'][contig][chunk_name][chunk]['position']
            position = np.array(position, dtype=np.int64)
            index = np.array(index, dtype=np.int)
            predictions = np.array(predictions, dtype=np.int)

            for pos, indx, pred in zip(position, index, predictions):
                if (pos, indx) not in prediction_dict:
                    prediction_dict[(pos, indx)] = pred
                    positions.add((pos, indx))

        pos_list = sorted(list(positions), key=lambda element: (element[0], element[1]))
        dict_fetch = operator.itemgetter(*pos_list)
        predicted_labels = list(dict_fetch(prediction_dict))
        sequence = ''.join([label_to_sequence(x) for x in predicted_labels])
        name_sequence_tuples.append((chunk_name, sequence))

    hdf5_file.close()

    return name_sequence_tuples


def get_confident_positions(alignment_a, alignment_b):
    match_counter = 0
    a_index = 0
    b_index = 0

    for base_a, base_b in zip(alignment_a, alignment_b):
        if base_a != '-':
            a_index += 1

        if base_b != '-':
            b_index += 1

        if base_a == base_b:
            match_counter += 1
        else:
            match_counter = 0

        if match_counter >= 3:
            return a_index, b_index

    return -1, -1


def create_consensus_sequence(hdf5_file_path, contig, sequence_chunk_keys, threads):
    chunk_name_to_sequence = defaultdict()

    # generate the dictionary in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
        file_chunks = chunks(sequence_chunk_keys, int(len(sequence_chunk_keys) / threads) + 1)

        futures = [executor.submit(small_chunk_stitch, hdf5_file_path, contig, file_chunk) for file_chunk in file_chunks]
        for fut in concurrent.futures.as_completed(futures):
            if fut.exception() is None:
                name_sequence_tuples = fut.result()
                for chunk_name, chunk_sequence in name_sequence_tuples:
                    chunk_name_to_sequence[chunk_name] = chunk_sequence
            else:
                sys.stderr.write("ERROR: " + str(fut.exception()) + "\n")
            fut._result = None  # python issue 27144

    print("DONE GENERATING THE CHUNK SEQUENCES")
    # but you cant do this part in parallel, this has to be linear
    chunk_names = sorted(sequence_chunk_keys)
    running_sequence = chunk_name_to_sequence[chunk_names[0]]
    running_start = int(chunk_names[0].split('-')[-2])
    running_end = int(chunk_names[0].split('-')[-1])

    if len(running_sequence) < 500:
        sys.stderr.write("ERROR: CURRENT SEQUENCE LENGTH TOO SHORT: " + sequence_chunk_keys[0] + "\n")
        exit()

    for i in range(1, len(chunk_names)):
        this_sequence = chunk_name_to_sequence[chunk_names[i]]
        this_start = int(chunk_names[i].split('-')[-2])
        this_end = int(chunk_names[i].split('-')[-1])

        if this_start < running_end:
            # overlap
            overlap_bases = running_end - this_start
            overlap_bases = overlap_bases + int(overlap_bases * BASE_ERROR_RATE)

            if overlap_bases > len(running_sequence):
                print("OVERLAP BASES ERROR WITH RUNNING SEQUENCE: ", overlap_bases, len(running_sequence))
            if overlap_bases > len(this_sequence):
                print("OVERLAP BASES ERROR WITH CURRENT SEQUENCE: ", overlap_bases, len(this_sequence))

            sequence_suffix = running_sequence[-overlap_bases:]
            sequence_prefix = this_sequence[:overlap_bases]
            alignments = pairwise2.align.globalxx(sequence_suffix, sequence_prefix)
            pos_a, pos_b = get_confident_positions(alignments[0][0], alignments[0][1])

            if pos_a == -1 or pos_b == -1:
                sys.stderr.write("ERROR: INVALID OVERLAPS: " + str(alignments[0]) + str(chunk_names[i])  + "\n")
                return None

            left_sequence = running_sequence[:-(overlap_bases-pos_a)]
            right_sequence = this_sequence[pos_b:]

            running_sequence = left_sequence + right_sequence
            running_end = this_end
        else:
            print("NO OVERLAP: POSSIBLE ERROR", chunk_names[i])

    sys.stderr.write("SUCCESSFULLY CALLED CONSENSUS SEQUENCE" + "\n")

    return running_sequence


def process_marginpolish_h5py(hdf_file_path, output_path, threads):
    hdf5_file = h5py.File(hdf_file_path, 'r')
    contigs = list(hdf5_file['predictions'].keys())

    consensus_fasta_file = open(output_path+'consensus.fa', 'w')
    for contig in contigs:
        chunk_keys = sorted(hdf5_file['predictions'][contig].keys())
        consensus_sequence = create_consensus_sequence(hdf_file_path, contig, chunk_keys, threads)
        if consensus_sequence is not None:
            consensus_fasta_file.write('>' + contig + "\n")
            consensus_fasta_file.write(consensus_sequence+"\n")

    hdf5_file.close()


if __name__ == '__main__':
    '''
    Processes arguments and performs tasks.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence_hdf",
        type=str,
        required=True,
        help="H5PY file generated by HELEN."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="CONSENSUS output directory."
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=5,
        help="Number of maximum threads for this region."
    )

    FLAGS, unparsed = parser.parse_known_args()
    process_marginpolish_h5py(FLAGS.sequence_hdf, FLAGS.output_dir, FLAGS.threads)
    # read_marginpolish_h5py(FLAGS.marginpolish_h5py_dir, FLAGS.output_h5py_dir, FLAGS.train_mode, FLAGS.threads)
