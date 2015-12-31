import numpy as np
import random
import os
import subprocess
import tempfile

from krotos.msd.processing import make_minibatch
from krotos.paths import PATHS
from krotos.exceptions import ErrorCallException
from krotos.debug import report


class Dataset(object):
    _initalized = False
    _shuffled = False
    _files = {
        'manifest': None,
        'linecount': None
    }

    @classmethod
    def _shuffle(cls):
        if cls._files['manifest']: cls._files['manifest'].close()
        cls._files['manifest'] = tempfile.NamedTemporaryFile()

        ret = subprocess.call(['shuf', '-o', cls._files['manifest'].name, PATHS['msd_tracks']])
        if ret: raise ErrorCallException(ret)

        cls._shuffled = True

        report("Million Song Dataset manifest shuffled.")

    @classmethod
    def _initalize(cls):
        p = subprocess.Popen(['wc', '-l', PATHS['msd_tracks']], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        std_out, std_err = p.communicate()
        if p.returncode: raise ErrorCallException(p.returncode)

        cls._files['linecount'] = int(std_out.strip().split()[0])

        cls._initalized = True

        report("Million Song Dataset manifest contains %d tracks." % cls._files['linecount'])

    def __init__(self, use_existing=False, training_split=0.7, validation_split=0.1, testing_split=0.2):
        if not self._initalized:
            self._initalize()

        if not (use_existing and self._load_dataset(lenient=True)):
            self._shuffle()
            self._split_dataset(training_split, validation_split, testing_split)

    def _split_dataset(self, training_split, validation_split, testing_split):
        total = float(training_split + validation_split + testing_split)
        cut_1 = int(self._files['linecount'] * ((training_split) / total))
        cut_2 = int(self._files['linecount'] * ((training_split + validation_split) / total))

        splits = {
            'training':     open(os.path.join(PATHS['sets_cache'], 'training_set.txt'), 'w'),
            'validation':   open(os.path.join(PATHS['sets_cache'], 'validation_set.txt'), 'w'),
            'testing':      open(os.path.join(PATHS['sets_cache'], 'testing_set.txt'), 'w')
        }

        with tempfile.NamedTemporaryFile() as tmp:
            p = subprocess.Popen(['head', '-n', str(cut_2), self._files['manifest'].name], stdout=tmp)
            if p.wait(): raise ErrorCallException(p.returncode)
            tmp.flush()

            p = subprocess.Popen(['head', '-n', str(cut_1), tmp.name], stdout=splits['training'])
            if p.wait(): raise ErrorCallException(p.returncode)

            p = subprocess.Popen(['tail', '-n', '+' + str(cut_1), tmp.name], stdout=splits['validation'])
            if p.wait(): raise ErrorCallException(p.returncode)

            p = subprocess.Popen(['tail', '-n', '+' + str(cut_2), self._files['manifest'].name], stdout=splits['testing'])
            if p.wait(): raise ErrorCallException(p.returncode)

        for f in splits.values():
            f.flush()
            f.close()

        split_percentages = (training_split / total * 100, validation_split / total * 100, testing_split / total * 100)
        report("Million Song Dataset split into %2.0f%% training, %2.0f%% validation, %2.0f%% testing." % split_percentages)

        self._load_dataset()

    def _load_dataset(self, lenient=False):
        try:
            self.files = {
                'training':     open(os.path.join(PATHS['sets_cache'], 'training_set.txt'), 'r'),
                'validation':   open(os.path.join(PATHS['sets_cache'], 'validation_set.txt'), 'r'),
                'testing':      open(os.path.join(PATHS['sets_cache'], 'testing_set.txt'), 'r')
            }

            report("Million Song Dataset loaded in %s." % PATHS['sets_cache'])
        except IOError as e:
            if lenient: return False
            raise e

        return True

    def get_minibatch(self, samples=10):
        minibatch = make_minibatch(samples, self.files['training'])
