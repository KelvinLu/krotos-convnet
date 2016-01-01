import numpy as np

from krotos.msd.utils import msd_hdf5
from krotos.msd.processing import make_minibatch
from krotos.debug import report



class Dataset(object):
    _initalized = False

    @classmethod
    def _initalize(cls):
        cls._sample_size = msd_hdf5.sample_size()

        cls._initalized = True

    def __init__(self, training_split=0.7, validation_split=0.1, testing_split=0.2):
        if not self._initalized:
            self._initalize()

        self._split_dataset(training_split, validation_split, testing_split)

    def _split_dataset(self, training_split, validation_split, testing_split):
        shuffle = np.random.permutation(self._sample_size)
        total = float(training_split + validation_split + testing_split)
        cut_1 = int(self._sample_size * ((training_split) / total))
        cut_2 = int(self._sample_size * ((training_split + validation_split) / total))
        self._training_inds     = shuffle[:cut_1]
        self._validation_inds   = shuffle[cut_1:cut_2]
        self._testing_inds      = shuffle[cut_2:]

    def _sample_training_ind(self, n=1):
        return np.random.choice(self._training_inds, size=n, replace=False)

    def minibatch(self, n=10):
        return make_minibatch(self, n)
