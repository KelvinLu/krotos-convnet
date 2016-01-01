import os



ROOT_PATH = os.path.dirname(os.path.realpath(__file__))

PATHS = {
    'msd_summary_h5':   os.path.join(ROOT_PATH, 'msd/resources/msd_summary_file.h5'),
    'msd_lastfm_db':    os.path.join(ROOT_PATH, 'msd/resources/lastfm_tags.db'),
    'tag_subset':       os.path.join(ROOT_PATH, 'msd/resources/tag_subset.pickle'),
}
