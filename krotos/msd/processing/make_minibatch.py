from multiprocessing.dummy import Pool as ThreadPool
import tempfile
import time
import datetime

from krotos.msd.utils import lastfm, latent, msd_hdf5, sevendigital
from krotos.msd.latent.features import LatentFeatures
from krotos.audio import spectrogram
from krotos.debug import report



WORKERS = 1



def make_minibatch(dataset, n=10, mapping='both', trim=False, normalize=False, audio_tempfile=False):
    remainder   = n
    results     = []

    pool    = ThreadPool(WORKERS)

    time_start_world    = time.time()
    time_start_proc     = time.clock()

    # Workers should never be processing tracks such that more than
    # n tracks are downloaded from 7digital. We must conserve our API calls.
    while remainder > 0:
        samples = select_samples(dataset, remainder, mapping, normalize, audio_tempfile)

        for success, result in pool.map(process_sample, samples):
            if success:
                results.append(result)
            elif result is not None:
                report("{0} {1}".format(datetime.datetime.now(), result))
        remainder = n - len(results)

        report("Minibatch: {}/{} samples downloaded and processed.".format(n - remainder, n), sameline=True)

    report("Minibatch: {} samples downloaded and processed in {}s ({}s process time).".format(n, time.time() - time_start_world, time.clock() - time_start_proc), sameline=True)

    if trim:
        if mapping == 'latent_features':
            results = [(sample['spectrogram_image'], sample['latent_features']) for sample in results]
        elif mapping == 'tag_vector':
            results = [(sample['spectrogram_image'], sample['tag_vector']) for sample in results]
        else:
            results = [(sample['spectrogram_image'], (sample['latent_features'], sample['tag_vector'])) for sample in results]

    return results

def select_samples(dataset, n, mapping='both', normalize=False, audio_tempfile=False):
    samples = []

    # Get metadata and Last.fm tags for a track.
    # Do sqlite database accesses single-threaded.
    while len(samples) < n:
        sample_ind          = dataset._sample_training_ind()
        track_id, metadata  = msd_hdf5.get_summary([sample_ind])

        track_id                                                    = track_id[0]
        track_id_7digital, track_id_echonest, title, artist_name    = metadata[0]

        if not track_id_7digital: continue

        latent_features = None
        if (mapping == 'both') or (mapping == 'latent_features'):
            latent_features = latent.get_latent_features(track_id_echonest, normalize)
            if latent_features is None: continue

        tag_vector  = None
        num_tags    = 0
        if (mapping == 'both') or (mapping == 'tag_vector'):
            tag_vector, num_tags = lastfm.get_tag_vector(track_id)
            if not num_tags: continue

        samples.append({
            'track_id':             track_id,
            'track_id_7digital':    track_id_7digital,
            'track_id_echonest':    track_id_echonest,
            'title':                title,
            'artist_name':          artist_name,
            'latent_features':      latent_features,
            'tag_vector':           tag_vector,
            'tempfile':             audio_tempfile,
        })

    return samples

def process_sample(sample):
    track_id_7digital = sample['track_id_7digital']

    f = tempfile.NamedTemporaryFile(suffix=".mp3")
    success, result = sevendigital.get_preview_track(track_id_7digital, f)
    if not success:
        f.close()
        return (False, result)

    f.flush()
    f.seek(0)

    success, result = spectrogram.mel_spectrogram(f.name)
    if not success:
        f.close()
        return (False, result)

    sample['spectrogram_image'] = result

    if sample['tempfile'] == True:
        sample['tempfile'] = f
    else:
        f.close()

    return (True, sample)
