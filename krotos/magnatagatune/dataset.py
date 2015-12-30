# Abstracts the notion of a training, validation, and testing dataset for the
# MagnaTagATune dataset.
#
# Provides functionality to read the .csv data from the dataset
#



import os
import numpy as np



ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
RESOURCES_DIR = 'resources'
ANNOTATIONS_FILE = 'annotations_final.csv'
MP3_DIR = 'mp3'



class _MagnaTagATuneDataset(type):
    _instances = {}
    _loaded = False

    def __call__(cls, *args, **kwargs):
        if not cls._loaded:
            cls._load_csv()

        return super(_MagnaTagATuneDataset, cls).__call__(*args, **kwargs)

    def _load_csv(cls):
        annotations_path = os.path.join(ROOT_PATH, RESOURCES_DIR, ANNOTATIONS_FILE)

        assert os.path.exists(annotations_path)

        with open(annotations_path, 'rb') as csv_tags:
            header = csv_tags.readline().rstrip().split(',')

            annotations = np.genfromtxt(csv_tags,
                delimiter=',',
                names=header,
                dtype='i4,' + 'i4,' * (len(header) - 2) + 'S256'
            )

        cls.fields      = list(annotations.dtype.names)[1:-1]
        cls._ids        = annotations[header[0]]
        cls._tags       = annotations[cls.fields]
        cls._mp3_paths  = annotations['mp3_path']

        cls.sample_size = annotations.size

        print('MagnaTagATune {0} read ({1} samples, {2} tags)'.
            format(ANNOTATIONS_FILE, cls.sample_size, len(cls.fields))
        )

        cls._loaded = True

    def _get_spectrogram(cls, sample_ind, file_cache=True, memory_cache=False):
        raise NotImplemented

class Dataset(object):
    __metaclass__ = _MagnaTagATuneDataset

    def __init__(self, training_split=0.7, validation_split=0.1, testing_split=0.3):
        self._split_dataset(training_split, validation_split, testing_split)

    def _split_dataset(self, training_split, validation_split, testing_split):
        shuffle = np.random.permutation(self.sample_size)
        total = float(training_split + validation_split + testing_split)
        cut_1 = int(self.sample_size * ((training_split) / total))
        cut_2 = int(self.sample_size * ((training_split + validation_split) / total))
        training_inds     = shuffle[:cut_1]
        validation_inds   = shuffle[cut_1:cut_2]
        testing_inds      = shuffle[cut_2:]

        self._training_ids          = self._ids[training_inds]
        self._validation_ids        = self._ids[validation_inds]
        self._testing_ids           = self._ids[testing_inds]

        self._training_tags         = self._tags[training_inds]
        self._validation_tags       = self._tags[validation_inds]
        self._testing_tags          = self._tags[testing_inds]

        self._training_mp3_paths    = self._mp3_paths[training_inds]
        self._validation_mp3_paths  = self._mp3_paths[validation_inds]
        self._testing_mp3_paths     = self._mp3_paths[testing_inds]

    def minibatch(self, samples=10):
        raise NotImplemented
