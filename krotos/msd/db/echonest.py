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

    def __init__(self):
        if not self._initialized:
            self._initialize()

    def get_size(self):
        m = self._execute(echonest.user_count()).fetchall()[0][0]
        n = self._execute(echonest.song_count()).fetchall()[0][0]

        return m, n

    def get_all_playcounts(self):
        # Remember that ROWIDs are 1-based indexes, not 0-based.
        # We will stick to the 0-based convention in this scope and expect
        # query function scopes to be 1-based.

        total = self._execute(echonest.triplet_count()).fetchall()[0][0]

        data = self._execute(echonest.get_triplet())

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

        data = self._execute(echonest.get_song_plays_by_user(user_rowid=(u + 1)))

        song_col_idxs, counts = zip(*data.fetchall())

        return song_col_idxs, counts
