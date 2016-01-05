def user_count(*args, **kwargs):
    return """SELECT count(*)
        FROM    vector_users;""".format(**kwargs)

def song_count(*args, **kwargs):
    return """SELECT count(*)
        FROM    vector_songs;""".format(**kwargs)

def triplet_count(*args, **kwargs):
    return """SELECT count(*)
        FROM    plays;""".format(**kwargs)

def get_triplet(*args, **kwargs):
    return """SELECT vector_users.ROWID - 1, vector_songs.ROWID - 1, plays.count
        FROM    vector_users, vector_songs, plays
        WHERE   vector_users.user = plays.user
        AND     vector_songs.song = plays.song;""".format(**kwargs)

def get_song_plays_by_user(*args, **kwargs):
    return """SELECT vector_songs.ROWID - 1, plays.count
        FROM    vector_users, vector_songs, plays
        WHERE   vector_users.ROWID = {user_rowid}
        AND     vector_users.user = plays.user
        AND     vector_songs.song = plays.song;""".format(**kwargs)
