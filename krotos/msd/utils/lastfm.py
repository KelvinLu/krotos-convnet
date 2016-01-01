from krotos.msd.db import LastFMTagsDB



TAG_DB = LastFMTagsDB()



def get_tag_vector(track_id):
    return TAG_DB.get_tag_vector(track_id)
    
def get_tag_human(track_id):
    return TAG_DB.get_tag_human(track_id)
