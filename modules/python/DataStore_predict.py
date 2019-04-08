import h5py
import yaml
import numpy as np
from modules.python.Options import ImageSizeOptions


class DataStore(object):
    """Class to read/write to a FRIDAY's file"""
    _prediction_path_ = 'predictions'
    _groups_ = ('position', 'index', 'bases', 'rles')

    def __init__(self, filename, mode='r'):
        self.filename = filename
        self.mode = mode

        self._sample_keys = set()
        self.file_handler = h5py.File(self.filename, self.mode)

        self._meta = None

    def __enter__(self):
        self.file_handler = h5py.File(self.filename, self.mode)

        return self

    def __exit__(self, *args):
        if self.mode != 'r' and self._meta is not None:
            self._write_metadata(self.meta)
        self.file_handler.close()

    def _write_metadata(self, data):
        """Save a data structure to file within a yml str."""
        for group, d in data.items():
            if group in self.file_handler:
                del self.file_handler[group]
            self.file_handler[group] = yaml.dump(d)

    def _load_metadata(self, groups=None):
        """Load meta data"""
        if groups is None:
            groups = self._groups_
        return {g: yaml.load(self.file_handler[g][()]) for g in groups if g in self.file_handler}

    @property
    def meta(self):
        if self._meta is None:
            self._meta = self._load_metadata()
        return self._meta

    def update_meta(self, meta):
        """Update metadata"""
        self._meta = self.meta
        self._meta.update(meta)

    def write_prediction(self, chromosome_name, chunk_name_prefix, chunk_name_suffix, position, index, bases, rles):
        self.file_handler['{}/{}/{}/{}/{}'.format(self._prediction_path_, chromosome_name, chunk_name_prefix,
                                                  chunk_name_suffix, 'position')] = position
        self.file_handler['{}/{}/{}/{}/{}'.format(self._prediction_path_, chromosome_name, chunk_name_prefix,
                                                  chunk_name_suffix, 'index')] = index
        self.file_handler['{}/{}/{}/{}/{}'.format(self._prediction_path_, chromosome_name, chunk_name_prefix,
                                                  chunk_name_suffix, 'bases')] = bases
        self.file_handler['{}/{}/{}/{}/{}'.format(self._prediction_path_, chromosome_name, chunk_name_prefix,
                                                  chunk_name_suffix, 'rles')] = rles


