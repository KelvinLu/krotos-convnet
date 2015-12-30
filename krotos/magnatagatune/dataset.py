# Abstracts the notion of a training, validation, and testing dataset for the
# MagnaTagATune dataset.
#
# Provides functionality to read the .csv data from the dataset
#



import os
import numpy as np



ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
RESOURCES_DIR = os.path.join(ROOT_PATH, 'resources')
SPECTROGRAM_DIR = os.path.join(RESOURCES_DIR, 'spectrogram_cache')
ANNOTATIONS_FILE = os.path.join(RESOURCES_DIR, 'annotations_final.csv')
MP3_DIR = os.path.join(RESOURCES_DIR, 'mp3')



class Dataset(object):
    _loaded = False

    @classmethod
    def _load_csv(cls):
        with open(ANNOTATIONS_FILE, 'rb') as csv_tags:
            header = csv_tags.readline().rstrip().split(',')
            header[0] = 'id'

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

    @classmethod
    def _get_spectrogram(cls, sample_ind, file_cache=True):
        sample_id = cls._ids[sample_ind]
        spectrogram_path = os.path.join(SPECTROGRAM_DIR, '{0}.mel' % sample_id)

        if file_cache and os.path.exists(spectrogram_path):
            # load cached spectrogram from file
            pass

        mp3_path = os.path.join(MP3_DIR, cls._mp3_paths[sample_ind])

        print mp3_path, os.path.exists(mp3_path)

        if file_cache:
            # save spectrogram
            pass

        raise NotImplemented

    def __init__(self, training_split=0.7, validation_split=0.1, testing_split=0.3):
        if not self._loaded:
            self._load_csv()

        self._split_dataset(training_split, validation_split, testing_split)

    def _split_dataset(self, training_split, validation_split, testing_split):
        shuffle = np.random.permutation(self.sample_size)
        total = float(training_split + validation_split + testing_split)
        cut_1 = int(self.sample_size * ((training_split) / total))
        cut_2 = int(self.sample_size * ((training_split + validation_split) / total))
        self._training_inds     = shuffle[:cut_1]
        self._validation_inds   = shuffle[cut_1:cut_2]
        self._testing_inds      = shuffle[cut_2:]

    def minibatch(self, samples=10):
        batch_inds = np.random.choice(self._training_inds, size=samples, replace=False)

        for ind in batch_inds:
            self._get_spectrogram(ind)

        raise NotImplemented

        return [], []
