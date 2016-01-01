from multiprocessing.dummy import Pool as ThreadPool
import tempfile

from krotos.msd.utils import msd_hdf5, sevendigital
from krotos.audio import spectrogram



WORKERS = 4



def make_minibatch(dataset, n=10):
    remainder   = n
    results     = []

    pool = ThreadPool(WORKERS)

    while remainder > 0:
        sample_inds = list(dataset._sample_training_ind(remainder))
        interim = pool.map(process_sample, sample_inds)

        results.extend([result for result in interim if result is not None])
        remainder = n - len(results)

    return results

def process_sample(sample_ind):
    track_id, metadata  = msd_hdf5.get_summary([sample_ind])

    track_id                                = track_id[0]
    track_id_7digital, title, artist_name   = metadata[0]

    with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
        success, response = sevendigital.get_preview_track(track_id_7digital, f)
        if not success: return None

        f.flush()
        f.seek(0)

        success, spec = spectrogram.mel_spectrogram(f.name)
        if not success: return None

    result = (spec,)
    return result
