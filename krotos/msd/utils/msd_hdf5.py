import os
import h5py
import random

from krotos.paths import PATHS



def random_track_id_h5(f):
    # http://stackoverflow.com/a/3540315
    #
    # The amazing Alex Martelli referencing the Resovoir Algorithm

    f.seek(0)
    line = next(f)
    for num, replace in enumerate(f):
        if random.randrange(num + 2): continue
        line = replace
    return line.split("<SEP>", 1)[0]

def metadata(track_id_h5):
    path = os.path.join(PATHS['msd_hdf5'], '/'.join(track_id_h5[2:5]), track_id_h5 + '.h5')

    with h5py.File(path, 'r') as f:
        meta = f['metadata']['songs'][0]

    return {
        'artist': meta['artist_name'],
        'title': meta['title'],
        'track_id_7digital': meta['track_7digitalid']
    }
