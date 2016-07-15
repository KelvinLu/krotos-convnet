import os
import errno



ROOT_PATH = os.path.dirname(os.path.realpath(__file__))

PATHS = {
    'msd_summary_h5':       os.path.join(ROOT_PATH, 'msd/resources/msd_summary_file.h5'),
    'msd_echonest_db':      os.path.join(ROOT_PATH, 'msd/resources/train_triplets.db'),
    'msd_echonest_latent':  os.path.join(ROOT_PATH, 'msd/resources/latent/'),
    'msd_lastfm_db':        os.path.join(ROOT_PATH, 'msd/resources/lastfm_tags.db'),
    'tag_subset':           os.path.join(ROOT_PATH, 'msd/resources/tag_subset.pickle'),
    'msd_dataset_split':    os.path.join(ROOT_PATH, 'msd/resources/split.pickle'),
    'convnet_dir':          os.path.join(ROOT_PATH, 'convnet/resources/'),
}

def mkdir_path(key):
    path = PATHS[key]
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise exc
