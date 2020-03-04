import os
import time


class FileManager:
    """
    Performs simple OS operations like creating directory for output or path to all the files.
    """
    @staticmethod
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

        return os.path.abspath(output_dir)

    @staticmethod
    def handle_train_output_directory(output_dir):
        """
        Process the output directory and return a valid directory where we save the output
        :param output_dir: Output directory path
        :return:
        """
        timestr = time.strftime("%m%d%Y_%H%M%S")
        # process the output directory
        if output_dir[-1] != "/":
            output_dir += "/"
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        # create an internal directory so we don't overwrite previous runs
        model_save_dir = output_dir + "trained_models_" + timestr + "/"
        if not os.path.exists(model_save_dir):
            os.mkdir(model_save_dir)

        stats_directory = model_save_dir + "stats_" + timestr + "/"

        if not os.path.exists(stats_directory):
            os.mkdir(stats_directory)

        return model_save_dir, stats_directory

    @staticmethod
    def get_file_paths_from_directory(directory_path):
        """
        Returns all paths of .h5 files given a directory path
        :param directory_path: Path to the directory
        :return: A list of paths of files
        """
        file_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path)
                      if os.path.isfile(os.path.join(directory_path, file)) and file[-2:] == 'h5']
        return file_paths

    @staticmethod
    def chunks(file_names, threads):
        """
        Given a list of file names a number of threads, this method returns a list containing file names.
        The len(chunks) == threads so we can use each item in the list for a single process.
        """
        chunks = []
        for i in range(0, len(file_names), threads):
            chunks.append(file_names[i:i + threads])
        return chunks

