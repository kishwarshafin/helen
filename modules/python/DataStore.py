import h5py
import yaml
import numpy as np


class DataStore(object):
    """
    This DataStore handles how we write an intermediate prediction file which we can use in stitch so we can
    do the stitching in parallel. The intended use of this object is to open under "with" statement.
    """
    # path to all the predictions in the HDF file.
    _prediction_path_ = 'predictions'
    # values that we save in the file
    _groups_ = ('position', 'index', 'bases', 'rles')

    def __init__(self, filename, mode='r'):
        """
        Object initialization function
        :param filename: Name of the function where to save the data
        :param mode: 'w' for write and 'r' for read.
        """
        # set the filename and mode
        self.filename = filename
        self.mode = mode

        self._sample_keys = set()
        # set the file handler
        self.file_handler = h5py.File(self.filename, self.mode)

        self._meta = None

    def __enter__(self):
        """
        This method is invoked when we open an object under "with" statement.
        :return:
        """
        self.file_handler = h5py.File(self.filename, self.mode)

        return self

    def __exit__(self, *args):
        """
        This method is invoked when the file object goes out of scope
        :param args:
        :return:
        """
        # we are not actively using the metadata now, we can remove this from the initial version
        if self.mode != 'r' and self._meta is not None:
            self.write_metadata(self.meta)
        self.file_handler.close()

    def write_metadata(self, data):
        """
        Save a data structure to file within a yml str.
        :param data: a dictionary containing metadata values
        """
        for group, d in data.items():
            if group in self.file_handler:
                del self.file_handler[group]
            self.file_handler[group] = yaml.dump(d)

    def load_metadata(self, groups=None):
        """
        Load meta data
        """
        if groups is None:
            groups = self._groups_
        return {g: yaml.load(self.file_handler[g][()]) for g in groups if g in self.file_handler}

    @property
    def meta(self):
        if self._meta is None:
            self._meta = self.load_metadata()
        return self._meta

    def update_meta(self, meta):
        """
        Update metadata
        """
        self._meta = self.meta
        self._meta.update(meta)

    def write_prediction(self, contig, contig_start, contig_end, chunk_id, position,
                         predicted_bases, predicted_rles, filename):
        """
        This is the method we use to write the data to the HDF file. This method is called by each image we
        generate.
        :param contig: Name of contig where the image belongs to.
        :param contig_start: Contig start position of the image.
        :param contig_end: Contig end position of the image.
        :param chunk_id: Chunk id from marginpolish to know which chunk the image is from.
        :param position: Array of values indicating genomic positions.
        :param predicted_bases: Array of values indicating predicted bases for each genomic position.
        :param predicted_rles: Array of values indicating predicted run-length for each genomic position.
        :param filename: Name of the file the image belongs to (used for debugging mostly)
        :return:
        """
        # get a chunk name prefix and suffix
        chunk_name_prefix = str(contig) + "-" + str(contig_start.item()) + "-" + str(contig_end.item())
        chunk_name_suffix = str(chunk_id.item())

        name = contig + chunk_name_prefix + chunk_name_suffix

        if 'predictions' not in self.meta:
            self.meta['predictions'] = set()
        if 'predictions_contig' not in self.meta:
            self.meta['predictions_contig'] = set()

        # the way we set up is predictions -> contig -> chunk_name_prefix -> chunk_name_suffix -> image
        # this way it's easier for stitch.py to grab all the images that belongs to a contig, split them by
        # chunk name prefix and then stitch the adjacent sequences together.

        # we set-up the contig -> chunk_name_prefix here. This sets up the contig start and end position for
        # marginpolish chunks
        if chunk_name_prefix not in self.meta['predictions_contig']:
            self.meta['predictions_contig'].add(chunk_name_prefix)
            self.file_handler['{}/{}/{}/{}'.format(self._prediction_path_, contig, chunk_name_prefix, 'contig_start')] \
                = contig_start.item()
            self.file_handler['{}/{}/{}/{}'.format(self._prediction_path_, contig, chunk_name_prefix, 'contig_end')] \
                = contig_end.item()

        # then we save the image predictions at the right place.
        if name not in self.meta['predictions']:
            self.meta['predictions'].add(name)
            self.file_handler['{}/{}/{}/{}/{}'.format(self._prediction_path_, contig, chunk_name_prefix,
                                                      chunk_name_suffix, 'position')] = np.array(position,
                                                                                                 dtype=np.uint32)
            self.file_handler['{}/{}/{}/{}/{}'.format(self._prediction_path_, contig, chunk_name_prefix,
                                                      chunk_name_suffix, 'bases')] = np.array(predicted_bases,
                                                                                              dtype=np.uint8)
            self.file_handler['{}/{}/{}/{}/{}'.format(self._prediction_path_, contig, chunk_name_prefix,
                                                      chunk_name_suffix, 'rles')] = np.array(predicted_rles,
                                                                                             dtype=np.uint8)
