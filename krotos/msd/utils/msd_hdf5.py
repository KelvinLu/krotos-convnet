import os
import h5py

from krotos.paths import PATHS



def metadata(h5_song):
    path = os.path.join(PATHS['msd_hdf5'], '/'.join(h5_song[2:5]), h5_song + '.h5')

    with h5py.File(path, 'r') as f:
        meta = f['metadata']['songs'][0]

    return {
        'artist': meta['artist_name'],
        'title': meta['title'],
        'track_id': meta['track_7digitalid']
    }
