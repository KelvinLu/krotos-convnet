import numpy as np
import os
import pickle

from krotos.paths import PATHS
from krotos.msd.utils import msd_hdf5
from krotos.msd.processing import make_minibatch
from krotos.debug import report



class Dataset(object):
    _initialized    = False
    _instance       = None

    @classmethod
    def _initialize(cls):
        cls._sample_size = msd_hdf5.sample_size()

        cls._initialized = True

    @classmethod
    def instance(cls, new=False, training_split=0.7, validation_split=0.1, testing_split=0.2):
        # Create a new split, regardless if one exists already.
        if new:
            cls._instance = cls(training_split=training_split, validation_split=validation_split, testing_split=testing_split)
            cls._save(cls._instance._serialize_to_instance())

        # If at runtime a split (now) exists, just return it.
        if cls._instance is not None:
            return cls._instance

        # Try to find saved split instance. Create and return a split from the
        # saved instance or from scratch.
        instance = cls._load()
        cls._instance = cls(instance=instance, training_split=training_split, validation_split=validation_split, testing_split=testing_split)

        # If no saved instance was found earlier, save the current one.
        if instance is None: cls._save(cls._instance._serialize_to_instance())

        return cls._instance

    @classmethod
    def _load(cls):
        if not os.path.isfile(PATHS['msd_dataset_split']): return None
        
        p = pickle.load(open(PATHS['msd_dataset_split'], 'rb'))

        cls._sample_size = p['sample_size']
        return p['split']

    @classmethod
    def _save(cls, instance):
        pickle.dump({
            'sample_size': cls._sample_size,
            'split': instance,
        }, open(PATHS['msd_dataset_split'], 'wb'))

    def __init__(self, training_split, validation_split, testing_split, instance=None):
        if instance is not None:
            self._init_from_instance(instance)
            self._initialized = True

        if not self._initialized:
            self._initialize()
            self._split_dataset(training_split, validation_split, testing_split)

        report("Million Dollar Dataset summary loaded.")

    def _init_from_instance(self, instance):
        self._training_inds     = instance['training_inds']
        self._validation_inds   = instance['validation_inds']
        self._testing_inds      = instance['testing_inds']

    def _serialize_to_instance(self):
        return {
            'training_inds':    self._training_inds,
            'validation_inds':  self._validation_inds,
            'testing_inds':     self._testing_inds,
        }

    def _split_dataset(self, training_split, validation_split, testing_split):
        shuffle = np.random.permutation(self._sample_size)
        total = float(training_split + validation_split + testing_split)
        cut_1 = int(self._sample_size * ((training_split) / total))
        cut_2 = int(self._sample_size * ((training_split + validation_split) / total))
        self._training_inds     = shuffle[:cut_1]
        self._validation_inds   = shuffle[cut_1:cut_2]
        self._testing_inds      = shuffle[cut_2:]

    def _sample_training_ind(self):
        return np.random.choice(self._training_inds)

    # TODO: add kwarg switch to create samples with Echo Nest latent feature
    # vectors (when that part is ready)

    def minibatch(self, n=10, trim=True):
        batch = make_minibatch(self, n)
        if trim:
            for s in batch: s[-1].close()
            batch = [(s[0], s[1]) for s in batch]

        return batch
