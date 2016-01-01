from multiprocessing.dummy import Pool as ThreadPool
import tempfile

from krotos.msd.utils import lastfm, msd_hdf5, sevendigital
from krotos.audio import spectrogram
from krotos.debug import report, report_newline



WORKERS = 4



def make_minibatch(dataset, n=10):
    remainder   = n
    results     = []

    pool    = ThreadPool(WORKERS)

    # Workers should never be processing tracks such that more than
    # n tracks are downloaded from 7digital. We must conserve our API calls.
    while remainder > 0:
        samples = make_samples_with_metadata(dataset, remainder)

        interim = pool.map(process_sample, samples)

        results.extend([result for result in interim if result is not None])
        remainder = n - len(results)

        report("Minibatch: {}/{} downloaded and processed.".format(n - remainder, n), sameline=True)

    report_newline()

    return results

def make_samples_with_metadata(dataset, n):
    results = []

    # Get Last.fm tags for a track.
    # Do sqlite database accesses single-threaded.
    while len(results) < n:
        sample_ind          = dataset._sample_training_ind()
        track_id, metadata  = msd_hdf5.get_summary([sample_ind])

        track_id                                = track_id[0]
        track_id_7digital, title, artist_name   = metadata[0]

        tag_vector, num_tags = lastfm.get_tag_vector(track_id)
        if not num_tags: continue

        results.append((sample_ind, tag_vector, track_id_7digital))

    return results

def process_sample(sample):
    sample_ind, tag_vector, track_id_7digital = sample
    with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
        success, response = sevendigital.get_preview_track(track_id_7digital, f)
        if not success: return None

        f.flush()
        f.seek(0)

        success, spec = spectrogram.mel_spectrogram(f.name)
        if not success: return None

    return (spec, tag_vector)

def examine_minibatch(dataset, n=10):
    remainder   = n
    results     = []

    while remainder > 0:
        samples = examine_samples_with_metadata(dataset, remainder)

        interim = map(examine_sample, samples)

        results.extend([result for result in interim if result is not None])
        remainder = n - len(results)

    return results

def examine_samples_with_metadata(dataset, n):
    results = []

    # Get Last.fm tags for a track
    # Do sqlite database accesses single-threaded.
    while len(results) < n:
        sample_ind          = dataset._sample_training_ind()
        track_id, metadata  = msd_hdf5.get_summary([sample_ind])

        track_id                                = track_id[0]
        track_id_7digital, title, artist_name   = metadata[0]

        tag_human, num_tags = lastfm.get_tag_human(track_id)
        if not num_tags: continue

        results.append((sample_ind, tag_human, track_id_7digital, title, artist_name))

    return results

def examine_sample(sample):
    sample_ind, tag_names, track_id_7digital, title, artist_name = sample
    f = tempfile.NamedTemporaryFile(suffix=".mp3")
    success, response = sevendigital.get_preview_track(track_id_7digital, f)
    if not success: return None

    f.flush()
    f.seek(0)

    success, spec = spectrogram.mel_spectrogram(f.name)
    if not success: return None

    report("{} - {} ({}) downloaded and processed.".format(artist_name, title, ', '.join(tag_names)))

    return (spec, tag_names, f, title, artist_name)
