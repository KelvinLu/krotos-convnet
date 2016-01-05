def most_popular_tags(*args, **kwargs):
    return """SELECT tags.ROWID, tags.tag, count(tid_tag.tag)
        FROM        tid_tag
        JOIN        tags
        ON          tid_tag.tag = tags.ROWID
        GROUP BY    tags.tag
        ORDER BY    count(tid_tag.tag) DESC
        LIMIT       {n};""".format(**kwargs)

def all_tags(*args, **kwargs):
    return """SELECT tid_tag.tag
        FROM    tid_tag, tids
        WHERE   tids.tid = '{track_id}'
        AND     tids.ROWID = tid_tag.tid;""".format(**kwargs)
