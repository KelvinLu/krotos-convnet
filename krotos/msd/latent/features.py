import os
import signal
import pickle
import numpy as np
from scipy import sparse

from krotos.paths import PATHS, mkdir_path
from krotos.utils import Singleton
from krotos.msd.db.echonest import EchoNestTasteDB
from krotos.exceptions import ParametersError
from krotos.debug import report
from krotos.msd.latent import als



LATENT_FEATURES = 50

LAMBDA  = 0.5
ALPHA   = 2.0
EPSILON = 1e6

SUBSET = False

mkdir_path('msd_echonest_latent')
STORE_FILES = {
    'shape':    'shape.pickle',
    'X':        'X.npy',
    'Y':        'Y.npy',
    'C':        'C.npz',
    'progress': 'progress.pickle'
}
GET_STORE_PATH = lambda x: os.path.join(PATHS['msd_echonest_latent'], ('subset_' if SUBSET else '') + STORE_FILES[x])



class LatentFeatures(object):
    __metaclass__ = Singleton

    def __init__(self):
        self._echonest = EchoNestTasteDB(subset=SUBSET)
        report("Echo Nest database connected.")

        self._get_hyperparams()

        self._load_latents()
        report("Latent matrices loaded.")

        self._load_confidence_matrix()
        report("Confidence matrix loaded.")

    def _get_hyperparams(self):
        # m: the number of users
        # n: the number of songs
        dims = self._load(GET_STORE_PATH('shape'))
        if dims is None:
            dims = self._echonest.get_size()
            self._save(GET_STORE_PATH('shape'), dims)

        self.m, self.n = dims

        # f: the number of latent features
        self.f = LATENT_FEATURES

    def _load(self, path, mode=None):
        if not os.path.exists(path): return None

        with open(path, 'rb') as f:
            if mode == 'ndarray':
                return np.load(f)
            elif mode == 'COO':
                obj = np.load(f)
                return sparse.coo_matrix(
                    (obj['data'], (obj['row'], obj['col'])),
                    shape=obj['shape']
                )
            else:
                return pickle.load(f)

    def _save(self, path, obj, mode=None):
        s = signal.signal(signal.SIGINT, signal.SIG_IGN)

        with open(path, 'wb') as f:
            if mode == 'ndarray':
                np.save(f, obj)
            elif mode == 'COO':
                np.savez(
                    f,
                    data=obj.data,
                    row=obj.row,
                    col=obj.col,
                    shape=obj.shape
                )
            else:
                pickle.dump(obj, f)

        signal.signal(signal.SIGINT, s)

    def _load_latents(self):
        # X: m-by-f matrix of user latent feature row vectors
        self.X = self._load(GET_STORE_PATH('X'), mode='ndarray')

        if self.X is None:
            self.X = np.random.rand(self.m, self.f) * 0.01
            self._save(GET_STORE_PATH('X'), self.X, mode='ndarray')

        # Y: n-by-f matrix of song latent feature row vectors
        self.Y = self._load(GET_STORE_PATH('Y'), mode='ndarray')

        if self.Y is None:
            self.Y = np.random.rand(self.n, self.f) * 0.01
            self._save(GET_STORE_PATH('Y'), self.Y, mode='ndarray')

    def _save_latents(self, mtx):
        if mtx == 'X': self._save(GET_STORE_PATH('X'), self.X, mode='ndarray')
        if mtx == 'Y': self._save(GET_STORE_PATH('Y'), self.Y, mode='ndarray')

    def _load_confidence_matrix(self):
        C = self._load(GET_STORE_PATH('C'), mode='COO')

        if C is None:
            C = als.confidence_transform(self._get_plays_matrix(mode='COO'), ALPHA, EPSILON)
            self._save(GET_STORE_PATH('C'), C, mode='COO')

        self.C_ui = C.tocsr()
        self.C_iu = C.transpose(copy=False).tocsr()

    def _get_plays_matrix(self, mode='COO'):
        # Load this data to generate confidence matrices and prediction vectors
        # in later computation. Always returns a COO matrix.

        if mode == 'LIL':
            return self._load_plays_matrix_LIL()
        if mode == 'COO':
            return self._load_plays_matrix_COO()

    def _load_plays_matrix_COO(self):
        report("COO mode: Extracting SQLite dump.")
        user, song, playcount = self._echonest.get_all_playcounts()

        report("COO mode: Extraction done, creating play count matrix.")
        return sparse.coo_matrix((playcount, (user, song)), shape=(self.m, self.n))

    def _load_plays_matrix_LIL(self):
        # LIL matrices are row-slice efficent, esp. when row indices are ordered
        # Get data by user incrementally
        R = sparse.lil_matrix((self.m, self.n))

        report("LIL mode: Extracting SQLite dump.")

        for u in xrange(self.m):
            song_col_idxs, counts = self._echonest.get_song_plays_by_user(u)
            R[u, song_col_idxs] = counts

            if (u % 100) == 0:
                report("{:7.3f}% of users read from SQLite.".format(u * 100.0 / self.m), sameline=True)

        return R.tocoo()

    def _confidence_row_iter(self, C_ab, A_row_ind):
        for i in xrange(C_ab.indptr[A_row_ind], C_ab.indptr[A_row_ind + 1]):
            yield C_ab.data[i], C_ab.indices[i]

    def _update_matrix(self, A, B, C_ab, param_lambda, start_ind=0, end_ind=None):
        rows        = A.shape[0]
        end_ind     = min(end_ind or rows, rows)
        batch_size  = end_ind - start_ind

        B_T_B_regularized =  B.T.dot(B) + (param_lambda * np.eye(self.f))

        for A_row_ind in xrange(start_ind, end_ind):
            self._update_vector(A, B, C_ab, B_T_B_regularized, A_row_ind)

            if A_row_ind % 100 == 0:
                report("{0:7.3f}% of latent feature matrix updated... ({1:.3f}% of batch complete)".format(A_row_ind * 100.0 / rows, (A_row_ind - start_ind) * 100.0 / batch_size), sameline=True)

    def _update_vector(self, A, B, C_ab, B_T_B_regularized, A_row_ind):
        left    = B_T_B_regularized.copy()
        right   = np.zeros(self.f)

        for confidence_ab, B_row_ind in self._confidence_row_iter(C_ab, A_row_ind):
            B_vector    = B[B_row_ind]
            left       += (confidence_ab - 1) * np.outer(B_vector, B_vector)
            right      += confidence_ab * B_vector

        A[A_row_ind]  = np.linalg.solve(left, right)

    def _load_progress(self):
        self.progress = self._load(GET_STORE_PATH('progress')) or {
            'rnd':  0,
            'mtx':  'X',
            'idx':  0
        }

    def _save_progress(self, **kwargs):
        self.progress.update(kwargs)
        self._save(GET_STORE_PATH('progress'), self.progress)

    def minimize(self, rounds=1, batch_size=20000):
        self._load_progress()

        for rnd in xrange(self.progress['rnd'], rounds):
            report("Round {} of minimization...".format(rnd + 1))

            if self.progress['mtx'] == 'X':
                report("Updating matrix X of user latent feature vectors.")
                while(self.progress['idx'] < self.m):
                    self._update_matrix(
                        self.X, self.Y, self.C_ui, LAMBDA,
                        start_ind   = self.progress['idx'],
                        end_ind     = self.progress['idx'] + batch_size
                    )
                    self._save_latents('X')
                    self._save_progress(idx=(self.progress['idx'] + batch_size))

                    report('')
                self._save_progress(mtx='Y', idx=0)

            if self.progress['mtx'] == 'Y':
                report("Updating matrix Y of song latent feature vectors.")
                while(self.progress['idx'] < self.n):
                    self._update_matrix(
                        self.Y, self.X, self.C_iu, LAMBDA,
                        start_ind   = self.progress['idx'],
                        end_ind     = self.progress['idx'] + batch_size
                    )
                    self._save_latents('Y')
                    self._save_progress(idx=(self.progress['idx'] + batch_size))

                    report('')
                self._save_progress(mtx='X', idx=0)

            self._save_progress(rnd=(rnd + 1))

    def get(self, track_id_echonest):
        idx = self._echonest.get_track_idx(track_id_echonest)
        if idx == None: return None, None
        return self.Y[idx, :], idx

    def closest(self, features, n=5, ordered=False):
        features_norm   = np.linalg.norm(features)
        song_norm       = np.linalg.norm(self.Y, axis=1)

        r = np.dot(self.Y, features) / (song_norm * features_norm)
        closest_idx = np.argpartition(r, -n)[-n:]

        track_ids_echonest, idxs = self._echonest.get_track_ids(closest_idx)
        idxs = list(idxs)
        results = zip(track_ids_echonest, r[idxs], song_norm[idxs])

        if ordered:
            results = sorted(results, key=lambda x: x[1], reverse=True)

        return results
