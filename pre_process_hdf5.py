import h5py
import argparse
import sys
from tqdm import tqdm
from os.path import isfile, join
from os import listdir
import concurrent.futures
from modules.python.DataStore import DataStore
from collections import defaultdict
import numpy as np
import operator


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


def decode_label(base, run_length):
    label_decoder = {'A': 0, 'C': 20, 'G': 40, 'T': 60, '_': 0}
    label = label_decoder[base]
    if base != '_':
        if int(run_length) == 0:
            print("LABELING ERROR: RUN LEGTH >1 WHEN BASE IS NOT GAP")
        label = label + int(run_length)

    return label


def file_reader_worker(file_names):
    gathered_values = []
    total_counts = defaultdict(int)
    for hdf5_filename in file_names:
        hdf5_file = h5py.File(hdf5_filename, 'r')

        chromosome_name = hdf5_filename.split('.')[-3].split("-")[0]

        image_dataset = hdf5_file['rleWeight']
        position_dataset = hdf5_file['position']

        image = np.array(image_dataset, dtype=np.float)

        label = []
        if 'label' in hdf5_file.keys():
            label_dataset = hdf5_file['label']
            for base, run_length in label_dataset:
                total_counts[decode_label(chr(base), run_length)] += 1
                label.append(decode_label(chr(base), run_length))

        position = []
        index = []
        for pos, indx in position_dataset:
            position.append(pos)
            index.append(indx)
        gathered_values.append((chromosome_name, image, position, index, label, hdf5_filename.split('/')[-1]))
    return gathered_values, total_counts


def merge_dicts(d, d1):
    '''
    https://stackoverflow.com/questions/31216001/how-to-concatenate-or-combine-two-defaultdicts-of-defaultdicts-that-have-overlap
    :param d:
    :param d1:
    :return:
    '''
    for k, v in d1.items():
        if k in d:
            d[k].update(d1[k])
        else:
            d[k] = d1[k]
    return d


def write_to_file_parallel(input_files, hdf5_output_file_name, threads):
    label_count = defaultdict(int)

    with DataStore(hdf5_output_file_name, 'w') as ds:
        with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
            file_chunks = chunks(input_files, min(64, int(len(input_files) / threads) + 1))
            futures = [executor.submit(file_reader_worker, file_chunk) for file_chunk in file_chunks]
            for fut in concurrent.futures.as_completed(futures):
                if fut.exception() is None:
                    gathered_values, label_dict = fut.result()
                    label_count.update(label_dict)

                    for chromosome_name, image, position, index, label, summary_name in gathered_values:
                        ds.write_train_summary(chromosome_name, image, position, index, label, summary_name)
                else:
                    sys.stderr.write(str(fut.exception()))
                fut._result = None  # python issue 27144

    print("THE LOSS-FUNCTION WEIGHTS SHOULD BE: ")
    print("[", end='')
    max_label, max_value = max(label_count.items(), key=operator.itemgetter(1))
    for i in range(0, 81):
        if i not in label_count.keys():
            print(float(max_value), end='')
        else:
            print(float(max_value) / float(label_count[i]), end='')

        if i < 80:
            print(', ', end='')
    print("]")


def process_marginpolish_h5py(marginpolish_output_directory, output_path, train_mode, threads):
    all_hdf5_file_paths = sorted(get_file_paths_from_directory(marginpolish_output_directory))

    if train_mode:
        total_length = len(all_hdf5_file_paths)
        selected_training_samples = int(total_length * 0.8)

        training_samples = all_hdf5_file_paths[:selected_training_samples]
        testing_samples = all_hdf5_file_paths[selected_training_samples:]

        train_data_file_name = output_path + "train" + "_images_marginpolish" + ".hdf"
        sys.stderr.write("WRITING " + str(len(training_samples)) + " SAMPLES TO train_images_marginpolish.hdf\n")

        write_to_file_parallel(training_samples, train_data_file_name, threads)

        test_data_file_name = output_path + "test" + "_images_marginpolish" + ".hdf"
        sys.stderr.write("WRITING " + str(len(testing_samples)) + " SAMPLES TO test_images_marginpolish.hdf\n")

        write_to_file_parallel(testing_samples, test_data_file_name, threads)
    else:
        train_data_file_name = output_path + "images_marginpolish" + ".hdf"
        sys.stderr.write("WRITING " + str(len(all_hdf5_file_paths)) + " SAMPLES TO images_marginpolish.hdf\n")
        write_to_file_parallel(all_hdf5_file_paths, train_data_file_name, threads)


def write_to_file(all_hdf5_file_paths, hdf5_data_file):
    for i in tqdm(range(0, len(all_hdf5_file_paths)), ncols=100):
        hdf5_filename = all_hdf5_file_paths[i]
        hdf5_file = h5py.File(all_hdf5_file_paths[i], 'r')
        label_decoder = {'A': 1, 'C': 2, 'G': 3, 'T': 4, '_': 0}

        chromosome_name = hdf5_filename.split('.')[-3].split("-")[0]
        pos_start = int(hdf5_filename.split('.')[-3].split('-')[1])

        image_dataset = hdf5_file['simpleWeight']
        position_dataset = hdf5_file['position']

        image = []
        for image_line in image_dataset:
            a_fwd, a_rev, c_fwd, c_rev, g_fwd, g_rev, t_fwd, t_rev, gap_fwd, gap_rev = list(image_line)
            image.append([a_fwd, c_fwd, g_fwd, t_fwd, a_rev, c_rev, g_rev, t_rev, gap_fwd, gap_rev])

        label = []
        if 'label' in hdf5_file.keys():
            label_dataset = hdf5_file['label']
            for l in label_dataset:
                label.append(label_decoder[chr(l[0])])

        position = []
        index = []
        for pos, indx in position_dataset:
            position.append(pos + pos_start)
            index.append(indx)
        hdf5_data_file.write_train_summary(chromosome_name, image, position, index, label, hdf5_filename.split('/')[-1])


def read_marginpolish_h5py(marginpolish_output_directory, output_path, train_mode, threads):
    all_hdf5_file_paths = sorted(get_file_paths_from_directory(marginpolish_output_directory))

    if train_mode:
        total_length = len(all_hdf5_file_paths)
        selected_training_samples = int(total_length * 0.8)

        training_samples = all_hdf5_file_paths[:selected_training_samples]
        testing_samples = all_hdf5_file_paths[selected_training_samples:]

        train_data_file_name = output_path + "train" + "_images_marginpolish" + ".hdf"
        sys.stderr.write("WRITING " + str(len(training_samples)) + "SAMPLES TO train_images_marginpolish.hdf")

        train_data_file = DataStore(train_data_file_name, mode='w')
        train_data_file.__enter__()
        write_to_file(training_samples, train_data_file)

        test_data_file_name = output_path + "test" + "_images_marginpolish" + ".hdf"
        sys.stderr.write("WRITING " + str(len(testing_samples)) + "SAMPLES TO test_images_marginpolish.hdf")

        test_data_file = DataStore(test_data_file_name, mode='w')
        test_data_file.__enter__()
        write_to_file(testing_samples, test_data_file)
    else:
        train_data_file_name = output_path + "images_marginpolish" + ".hdf"

        sys.stderr.write("WRITING " + str(len(all_hdf5_file_paths)) + "SAMPLES TO images_marginpolish.hdf")
        train_data_file = DataStore(train_data_file_name, mode='w')
        train_data_file.__enter__()
        write_to_file(all_hdf5_file_paths, train_data_file)


if __name__ == '__main__':
    '''
    Processes arguments and performs tasks.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--marginpolish_h5py_dir",
        type=str,
        required=True,
        help="H5PY file generated by HELEN."
    )
    parser.add_argument(
        "--output_h5py_dir",
        type=str,
        required=True,
        help="H5PY file generated by MEDAKA."
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=5,
        help="Number of maximum threads for this region."
    )
    parser.add_argument(
        "--train_mode",
        type=lambda x: (str(x).lower() == 'true' or str(x).lower() == '1'),
        default=False,
        help="If true then compare labels too."
    )

    FLAGS, unparsed = parser.parse_known_args()
    process_marginpolish_h5py(FLAGS.marginpolish_h5py_dir, FLAGS.output_h5py_dir, FLAGS.train_mode, FLAGS.threads)
    # read_marginpolish_h5py(FLAGS.marginpolish_h5py_dir, FLAGS.output_h5py_dir, FLAGS.train_mode, FLAGS.threads)
