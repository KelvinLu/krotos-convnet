from krotos.msd.db import LastFMTagsDB



def get_tag_vector(track_id):
    return LastFMTagsDB().get_tag_data(track_id)

def get_tag_names(tag_vector):
    return LastFMTagsDB().get_tag_names(tag_vector)
