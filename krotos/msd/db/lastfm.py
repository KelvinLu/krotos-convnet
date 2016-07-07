import os
import pickle
import numpy as np

from krotos.paths import PATHS
from krotos.utils import Singleton
from krotos.msd.db.dbbase import DBConn
from krotos.msd.db.queries import lastfm
from krotos.debug import report



N_TAGS = 200



class LastFMTagsDB(DBConn):
    __metaclass__ = Singleton

    _initialized    = False
    _tag_subset     = None
    _tag_subset_ids = None

    @classmethod
    def _initialize(cls):
        cls._establish_db_conn(PATHS['msd_lastfm_db'])
        cls._load_tag_subset()
        cls._tag_subset_ids = np.sort([tag[0] for tag in cls._tag_subset])

        cls._initialized = True

    @classmethod
    def _load_tag_subset(cls):
        if os.path.exists(PATHS['tag_subset']):
            with open(PATHS['tag_subset'], 'r') as f:
                cls._tag_subset = pickle.load(f)
            return

        res = cls._execute(lastfm.most_popular_tags(n=N_TAGS))
        cls._tag_subset = sorted(res.fetchall(), key=lambda row: row[0])

        with open(PATHS['tag_subset'], 'w') as f:
            pickle.dump(cls._tag_subset, f)

    def __init__(self):
        if not self._initialized:
            self._initialize()

        report("Last.fm database connected.")

    def get_tag_data(self, track_id):
        tags = self.get_tag_ids(track_id)

        return (np.vectorize(lambda x: 1.0 if x in tags else 0.0)(self._tag_subset_ids),
            len(tags)
        )

    def get_tag_ids(self, track_id):
        res = self._execute(lastfm.all_tags(track_id=track_id))
        data = res.fetchall()

        return np.intersect1d([row[0] for row in data], self._tag_subset_ids)

    def get_tag_names(self, tag_vector):
        if tag_vector is None: return []
        return [row[1] for i, row in zip(xrange(N_TAGS), self._tag_subset) if tag_vector[i] > 0.0]
