import h5py
import numpy as np

from krotos.paths import PATHS
from krotos.debug import report



SUMMARY_HANDLE = h5py.File(PATHS['msd_summary_h5'], 'r')

# inds must be sorted for the HDF5 reader
def get_summary(inds):
    track_id = SUMMARY_HANDLE['analysis']['songs'][inds]['track_id']
    metadata = SUMMARY_HANDLE['metadata']['songs'][inds][['track_7digitalid', 'song_id', 'title', 'artist_name']]

    return track_id, metadata

def sample_size():
    return SUMMARY_HANDLE['analysis']['songs'].size
