import os



ROOT_PATH = os.path.dirname(os.path.realpath(__file__))

PATHS = {
    'msd_hdf5':             os.path.join(ROOT_PATH, 'msd/resources/MillionSongSubset/data'),
    'msd_tracks':           os.path.join(ROOT_PATH, 'msd/resources/MillionSongSubset/AdditionalFiles/subset_unique_tracks.txt'),
    'sets_cache':           os.path.join(ROOT_PATH, 'msd/resources/sets_cache'),
    'spectrogram_cache':    os.path.join(ROOT_PATH, 'msd/resources/spectrogram_cache')
}
