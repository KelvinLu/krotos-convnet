import sqlite3



class DBConn(object):
    _conn           = None

    @classmethod
    def _initialize(cls):
        cls._establish_db_conn()

    @classmethod
    def _execute(cls, query):
        return cls._conn.execute(query)

    @classmethod
    def _commit(cls):
        cls._conn.commit()

    @classmethod
    def _establish_db_conn(cls, path):
        cls._conn = sqlite3.connect(path)
