from multiprocessing.dummy import Pool as ThreadPool
import tempfile

from krotos.msd.utils import lastfm, msd_hdf5, sevendigital
from krotos.audio import spectrogram
from krotos.debug import report, report_newline



WORKERS = 4



# TODO: refactor code minibatch generation to create samples for Echo Nest
# latent feature vectors (when that part is ready)



def make_minibatch(dataset, n=10):
    remainder   = n
    results     = []

    pool    = ThreadPool(WORKERS)

    # Workers should never be processing tracks such that more than
    # n tracks are downloaded from 7digital. We must conserve our API calls.
    while remainder > 0:
        samples = make_sample_tuples(dataset, remainder)

        interim = pool.map(process_sample, samples)

        results.extend([result for result in interim if result is not None])
        remainder = n - len(results)

        report("Minibatch: {}/{} downloaded and processed.".format(n - remainder, n), sameline=True)

    report_newline()

    return results

def make_sample_tuples(dataset, n):
    results = []

    # Get metadata and Last.fm tags for a track.
    # Do sqlite database accesses single-threaded.
    while len(results) < n:
        sample_ind          = dataset._sample_training_ind()
        track_id, metadata  = msd_hdf5.get_summary([sample_ind])

        track_id                                                    = track_id[0]
        track_id_7digital, track_id_echonest, title, artist_name    = metadata[0]

        if not track_id_7digital: continue

        tag_vector, tag_names, num_tags = lastfm.get_tag_data(track_id)
        if not num_tags: continue

        results.append((track_id_7digital, title, artist_name, tag_vector, tag_names, num_tags))

    return results

def process_sample(sample):
    track_id_7digital, title, artist_name, tag_vector, tag_names, num_tags = sample

    f = tempfile.NamedTemporaryFile(suffix=".mp3")
    success, response = sevendigital.get_preview_track(track_id_7digital, f)
    if not success: return None

    f.flush()
    f.seek(0)

    success, spec = spectrogram.mel_spectrogram(f.name)
    if not success: return None

    return (spec, tag_vector, title, artist_name, tag_names, num_tags, f)
