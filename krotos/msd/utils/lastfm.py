from krotos.msd.db import LastFMTagsDB



TAG_DB = LastFMTagsDB()



def get_tag_data(track_id):
    return TAG_DB.get_tag_data(track_id)
