import os
import pickle
import numpy as np
from scipy import sparse

from krotos.paths import PATHS, mkdir_path
from krotos.utils import Singleton
from krotos.msd.db.echonest import EchoNestTasteDB
from krotos.exceptions import ParametersError
from krotos.debug import report, report_newline



LATENT_FEATURES = 50

LAMBDA  = 0.05
ALPHA   = 2.0
EPSILON = 1e6

SUBSET = True

mkdir_path('msd_echonest_latent')
STORE_FILES = {
    'shape':    'shape.pickle',
    'X':        'X.npy',
    'Y':        'Y.npy',
    'r':        'r.npz',
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

        self._load_plays_matrix()
        report("Play count matrix loaded.")

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
            elif mode == 'CSR':
                obj = np.load(f)
                return sparse.csr_matrix(
                    (obj['data'], obj['indices'], obj['indptr']),
                    shape=obj['shape']
                )
            else:
                return pickle.load(f)

    def _save(self, path, obj, mode=None):
        with open(path, 'wb') as f:
            if mode == 'ndarray':
                np.save(f, obj)
            elif mode == 'CSR':
                np.savez(
                    f,
                    data=obj.data,
                    indices=obj.indices,
                    indptr=obj.indptr,
                    shape=obj.shape
                )
            else:
                pickle.dump(obj, f)

    def _load_latents(self):
        # X: m-by-f matrix of user latent feature row vectors
        self.X = self._load(GET_STORE_PATH('X'), mode='ndarray')

        if self.X is None:
            self.X = np.random.rand(self.m, self.f)
            self._save(GET_STORE_PATH('X'), self.X, mode='ndarray')

        # Y: n-by-f matrix of song latent feature row vectors
        self.Y = self._load(GET_STORE_PATH('Y'), mode='ndarray')

        if self.Y is None:
            self.Y = np.random.rand(self.n, self.f)
            self._save(GET_STORE_PATH('Y'), self.Y, mode='ndarray')

    def _save_latents(self, mtx):
        if mtx == 'X': self._save(GET_STORE_PATH('X'), self.X, mode='ndarray')
        if mtx == 'Y': self._save(GET_STORE_PATH('Y'), self.Y, mode='ndarray')

    def _load_plays_matrix(self, mode='COO'):
        # Load this data to generate confidence matrices and prediction vectors
        # in later computation

        self.r = self._load(GET_STORE_PATH('r'), mode='CSR')

        if self.r is None:
            if mode == 'LIL':
                self.r = self._load_plays_matrix_LIL().tocsr()
            if mode == 'COO':
                self.r = self._load_plays_matrix_COO().tocsr()
            self._save(GET_STORE_PATH('r'), self.r, mode='CSR')

        self.r_T = self.r.transpose(copy=False).tocsr()

    def _load_plays_matrix_LIL(self):
        # LIL matrices are row-slice efficent, esp. when row indices are ordered
        # Get data by user incrementally
        r = sparse.lil_matrix((self.m, self.n))

        report("LIL mode: Extracting SQLite dump.")

        for u in xrange(self.m):
            song_col_idxs, counts = self._echonest.get_song_plays_by_user(u)
            r[u, song_col_idxs] = counts

            if (u % 100) == 0:
                report("{:7.3f}% of users read from SQLite.".format(u * 100.0 / self.m), sameline=True)

        report_newline()
        return r.tocoo()

    def _load_plays_matrix_COO(self):
        report("COO mode: Extracting SQLite dump.")
        user, song, playcount = self._echonest.get_all_playcounts()

        report("COO mode: Extraction done, creating play count matrix.")
        return sparse.coo_matrix((playcount, (user, song)), shape=(self.m, self.n))

    def _prediction(self, d, indptr_start, indptr_stop, inds):
        nnz             = indptr_stop - indptr_start
        data            = np.ones(nnz)
        indptr          = np.array([0, nnz])
        indices         = inds

        return sparse.csr_matrix((data, indices, indptr), shape=(1, d))

    def _confidence(self, d, indptr_start, indptr_stop, inds):
        # Avoid initializing a CSR matrix when referencing a row of self.r/r_T.
        r_vector = self.r.data[indptr_start:indptr_stop]

        c_minus_one = ALPHA * np.log(1 + EPSILON * r_vector)
        # C_minus_eye has a sparse diagonal, we construct it with a COO matrix
        C_minus_eye = sparse.coo_matrix((c_minus_one, (inds, inds)), shape=(d, d))
        # Faster than C = sparse.identity(d); C.data[0][inds] += c_minus_one
        C = sparse.coo_matrix((c_minus_one + 1, (inds, inds)), shape=(d, d))

        return C.tocsr(), C_minus_eye.tocsr()

    def _update_X(self, start_u=0, end_u=None):
        end         = min(end_u or self.m, self.m)
        batch_size  = end - start_u

        Y_T                                     = self.Y.T
        Y_T_Y_regularized                       = np.dot(Y_T, self.Y)
        # Just as fast as the .ravel() trick and fill_diagonal() methods.
        Y_T_Y_regularized.flat[::self.m + 1]   += LAMBDA
        for u in xrange(start_u, end):
            self._update_X_u(u, Y_T, Y_T_Y_regularized)

            if u % 10 == 0:
                report("{0:7.3f}% of X updated... ({1:.3f}% of batch complete)".format(u * 100.0 / self.m, (u - start_u) * 100.0 / batch_size), sameline=True)

    def _update_X_u(self, u, Y_T, Y_T_Y_regularized):
        d               = self.n
        indptr_start    = self.r.indptr[u]
        indptr_stop     = self.r.indptr[u+1]
        inds            = self.r.indices[indptr_start:indptr_stop]

        C_u, C_u_minus_eye = self._confidence(d, indptr_start, indptr_stop, inds)

        inner = C_u_minus_eye.dot(self.Y)
        inner = np.dot(Y_T, inner) + Y_T_Y_regularized

        left = np.linalg.inv(inner)
        left_T = np.dot(self.Y, left.T)

        right_T = self._prediction(d, indptr_start, indptr_stop, inds).dot(C_u)

        X_u = right_T.dot(left_T)

        self.X[u, :] = X_u

    def _update_Y(self, start_i=0, end_i=None):
        end         = min(end_i or self.n, self.n)
        batch_size  = end - start_i

        X_T                                     = self.X.T
        X_T_X_regularized                       = np.dot(X_T, self.X)
        # Just as fast as the .ravel() trick and fill_diagonal() methods.
        X_T_X_regularized.flat[::self.n + 1]   += LAMBDA

        for i in xrange(start_i, end):
            self._update_Y_i(i, X_T, X_T_X_regularized)

            if i % 10 == 0:
                report("{0:7.3f}% of Y updated... ({1:.3f}% of batch complete)".format(i * 100.0 / self.n, (i - start_i) * 100.0 / batch_size), sameline=True)

    def _update_Y_i(self, i, X_T, X_T_X_regularized):
        d               = self.m
        indptr_start    = self.r_T.indptr[i]
        indptr_stop     = self.r_T.indptr[i+1]
        inds            = self.r_T.indices[indptr_start:indptr_stop]

        C_i, C_i_minus_eye = self._confidence(d, indptr_start, indptr_stop, inds)

        inner = C_i_minus_eye.dot(self.X)
        inner = np.dot(X_T, inner) + X_T_X_regularized

        left = np.linalg.inv(inner)
        left_T = np.dot(self.X, left.T)

        right_T = self._prediction(d, indptr_start, indptr_stop, inds).dot(C_i)

        Y_i = right_T.dot(left_T)

        self.Y[i, :] = Y_i

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
                    self._update_X(
                        start_u = self.progress['idx'],
                        end_u   = self.progress['idx'] + batch_size
                    )
                    self._save_latents('X')
                    self._save_progress(idx=(self.progress['idx'] + batch_size))

                report_newline()
                self._save_progress(mtx='Y', idx=0)

            if self.progress['mtx'] == 'Y':
                report("Updating matrix Y of song latent feature vectors.")
                while(self.progress['idx'] < self.n):
                    self._update_Y(
                        start_i = self.progress['idx'],
                        end_i   = self.progress['idx'] + batch_size
                    )
                    self._save_latents('Y')
                    self._save_progress(idx=(self.progress['idx'] + batch_size))

                report_newline()
                self._save_progress(mtx='X', idx=0)

            self._save_progress(rnd=(rnd + 1))

    def get(self, track_id_echonest):
        idx = self._echonest.get_track_idx(track_id_echonest)
        return self.Y[idx, :], idx

    def closest(self, features, n=5):
        features_norm   = np.linalg.norm(features)
        song_norm       = np.linalg.norm(self.Y, axis=1)

        r = np.dot(self.Y, features) / (song_norm * features_norm)
        closest_idx = np.argpartition(r, -n)[-n:]

        track_ids_echonest, idxs = self._echonest.get_track_ids(closest_idx)

        return track_ids_echonest, r[list(idxs)]
