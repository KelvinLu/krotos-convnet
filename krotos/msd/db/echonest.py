import numpy as np

from krotos.paths import PATHS
from krotos.msd.db.dbbase import DBConn
from krotos.msd.db.queries import echonest
from krotos.debug import report, report_newline



def generator_result(cursor, size=1000):
    while True:
        batch = cursor.fetchmany(size)
        if not batch:
            break
        yield batch

class EchoNestTasteDB(DBConn):
    _initialized    = False

    @classmethod
    def _initialize(cls):
        cls._establish_db_conn(PATHS['msd_echonest_db'])

        cls._initialized = True

    def __init__(self, subset=True):
        if not self._initialized:
            self._initialize()

        self.subset = subset

    def get_size(self):
        m = self._execute(echonest.user_count(subset=self.subset)).fetchall()[0][0]
        n = self._execute(echonest.song_count(subset=self.subset)).fetchall()[0][0]

        return m, n

    def get_all_playcounts(self):
        # Remember that ROWIDs are 1-based indexes, not 0-based.
        # We will stick to the 0-based convention in this scope and expect
        # query function scopes to be 1-based.

        total = self._execute(echonest.triplet_count(subset=self.subset)).fetchall()[0][0]

        data = self._execute(echonest.get_triplet(subset=self.subset))

        result = np.ndarray(shape=(0,3))

        for batch in generator_result(data, size=10000):
            result = np.append(result, batch, axis=0)
            report("{:7.3f}% of rows fetched.".format(result.shape[0] * 100.0 / total), sameline=True)

        report_newline()

        return result[:, 0], result[:, 1], result[:, 2]

    def get_song_plays_by_user(self, u):
        # Remember that ROWIDs are 1-based indexes, not 0-based.
        # We will stick to the 0-based convention in this scope and expect
        # query function scopes to be 1-based.

        data = self._execute(echonest.get_song_plays_by_user(subset=self.subset, user_rowid=(u + 1)))

        song_col_idxs, counts = zip(*data.fetchall())

        return song_col_idxs, counts

    def get_track_idx(self, track_id_echonest):
        result_set = self._execute(echonest.get_track_idx(subset=self.subset, track_id_echonest=track_id_echonest)).fetchall()
        if not len(result_set): return None
        return result_set[0][0]

    def get_track_id(self, idx):
        return self._execute(echonest.get_track_id(subset=self.subset, idx=(idx + 1))).fetchall()[0][0]

    def get_track_ids(self, idxs):
        data = self._execute(echonest.get_track_ids(subset=self.subset, idxs_string=(', '.join(str(idx + 1) for idx in idxs)))).fetchall()
        track_ids_echonest, idxs = zip(*data)

        return track_ids_echonest, idxs
