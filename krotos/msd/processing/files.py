import os
import tempfile

from krotos.msd.utils import msd_hdf5, sevendigital
from krotos.audio import spectrogram



def get_spectrogram(track_id_h5):
    with tempfile.NamedTemporaryFile() as f:
        meta = msd_hdf5.metadata(track_id_h5)
        sevendigital.get_preview_track(meta['track_id_7digital'], f)

        return spectrogram.mel_spectrogram(f.name)
