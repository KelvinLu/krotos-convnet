# Abstracts the notion of a training, validation, and testing dataset for the
# MagnaTagATune dataset.
#
# Provides functionality to read the .csv data from the dataset
#



import os
import numpy as np

from krotos.audio.spectrogram import mel_spectrogram
from krotos.debug import report



ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
RESOURCES_DIR = os.path.join(ROOT_PATH, 'resources')
SPECTROGRAM_DIR = os.path.join(RESOURCES_DIR, 'spectrogram_cache')
ANNOTATIONS_FILE = os.path.join(RESOURCES_DIR, 'annotations_final.csv')
SPLITS_FILE = os.path.join(RESOURCES_DIR, 'splits.npz')
MP3_DIR = os.path.join(RESOURCES_DIR, 'mp3')



class Dataset(object):
    _loaded = False

    @classmethod
    def _load_csv(cls):
        # Read .csv file
        with open(ANNOTATIONS_FILE, 'rb') as csv_tags:
            header = csv_tags.readline().rstrip().split(',')
            header[0] = 'id'

            annotations = np.genfromtxt(csv_tags,
                delimiter=',',
                names=header,
                dtype='i4,' + 'i4,' * (len(header) - 2) + 'S256'
            )

        # Get each tag's name
        cls.fields      = list(annotations.dtype.names)[1:-1]

        # Trim dataset samples that don't have any tags
        tags_ndarray = annotations[cls.fields].view(('i4', len(cls.fields)))
        has_tags = np.sum(tags_ndarray, axis=1) > 0
        annotations = annotations[has_tags]

        # Get slices for each part of the dataset; ids, tags, and mp3 paths
        cls._ids        = annotations[header[0]]
        cls._tags       = annotations[cls.fields]
        cls._mp3_paths  = annotations['mp3_path']

        cls.sample_size = annotations.size

        report('MagnaTagATune {0} read ({1} samples, {2} tags)'.
            format(ANNOTATIONS_FILE, cls.sample_size, len(cls.fields))
        )

        cls._loaded = True

    @classmethod
    def _get_spectrogram(cls, sample_ind, file_cache=True):
        sample_id = cls._ids[sample_ind]
        spectrogram_path = os.path.join(SPECTROGRAM_DIR, '{0}.npy'.format(sample_id))

        if file_cache and os.path.exists(spectrogram_path):
            f = open(spectrogram_path, mode='r')
            spectrogram = np.load(f)
            f.close()

        mp3_path = os.path.join(MP3_DIR, cls._mp3_paths[sample_ind])
        assert os.path.exists(mp3_path)

        success, spectrogram = mel_spectrogram(mp3_path, duration=29.0)

        assert success

        if file_cache:
            f = open(spectrogram_path, mode='w')
            np.save(f, spectrogram)
            f.close()

        return spectrogram

    def __init__(self, new=False, training_split=0.7, validation_split=0.1, testing_split=0.2):
        if not self._loaded:
            self._load_csv()

        self._split_dataset(new, training_split, validation_split, testing_split)

    def _split_dataset(self, new=False, training_split=0.7, validation_split=0.1, testing_split=0.2):
        if (not new and os.path.isfile(SPLITS_FILE)):
            with open(SPLITS_FILE, 'rb') as f:
                npz = np.load(f)
                self._training_inds     = npz['training_inds']
                self._validation_inds   = npz['validation_inds']
                self._testing_inds      = npz['testing_inds']
            return

        shuffle = np.random.permutation(self.sample_size)
        total = float(training_split + validation_split + testing_split)
        cut_1 = int(self.sample_size * ((training_split) / total))
        cut_2 = int(self.sample_size * ((training_split + validation_split) / total))
        self._training_inds     = shuffle[:cut_1]
        self._validation_inds   = shuffle[cut_1:cut_2]
        self._testing_inds      = shuffle[cut_2:]
        with open(SPLITS_FILE, 'wb') as f:
            np.savez(f, training_inds=self._training_inds, validation_inds=self._validation_inds, testing_inds=self._testing_inds)

    def minibatch(self, samples=10):
        batch_inds = np.random.choice(self._training_inds, size=samples, replace=False)

        return [(self._get_spectrogram(ind), np.asarray(list(self._tags[ind]), dtype='f4')) for ind in batch_inds]

    def human_examine(self, samples=5):
        batch_inds = np.random.choice(self.sample_size, size=samples, replace=False)

        for ind in batch_inds:
            yield (self._get_spectrogram(ind),
                [field for field, tag in zip(self.fields, self._tags[ind]) if tag],
                os.path.join(MP3_DIR, self._mp3_paths[ind])
            )
