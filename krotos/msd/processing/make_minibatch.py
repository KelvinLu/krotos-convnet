from multiprocessing.dummy import Pool as ThreadPool
import tempfile
import time

from krotos.msd.utils import lastfm, latent, msd_hdf5, sevendigital
from krotos.msd.latent.features import LatentFeatures
from krotos.audio import spectrogram
from krotos.debug import report, report_newline



WORKERS = 4



def make_minibatch(dataset, n=10, mapping='BOTH', trim=False, audio_tempfile=False):
    remainder   = n
    results     = []

    pool    = ThreadPool(WORKERS)

    time_start_world    = time.time()
    time_start_proc     = time.clock()

    # Workers should never be processing tracks such that more than
    # n tracks are downloaded from 7digital. We must conserve our API calls.
    while remainder > 0:
        samples = select_samples(dataset, remainder, mapping, audio_tempfile)

        interim = pool.map(process_sample, samples)

        results.extend([result for result in interim if result is not None])
        remainder = n - len(results)

        report("Minibatch: {}/{} samples downloaded and processed.".format(n - remainder, n), sameline=True)

    report("Minibatch: {} samples downloaded and processed in {}s ({}s process time).".format(n, time.time() - time_start_world, time.clock() - time_start_proc), sameline=True)
    report_newline()

    # if trim:
    #     results = [(sample['spectrogram_image'], sample['mapping']) for sample in results]

    return results

def select_samples(dataset, n, mapping='BOTH', audio_tempfile=False):
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
        if (mapping == 'BOTH') or (mapping == 'LATENT_FEATURES'):
            latent_features                 = latent.get_latent_features(track_id_echonest)
            if latent_features is None: continue

        tag_vector  = None
        num_tags    = 0
        if (mapping == 'BOTH') or (mapping == 'TAG_VECTOR'):
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
    success, response = sevendigital.get_preview_track(track_id_7digital, f)
    if not success:
        f.close()
        return None

    f.flush()
    f.seek(0)

    success, spec = spectrogram.mel_spectrogram(f.name)
    if not success:
        f.close()
        return None

    sample['spectrogram_image'] = spec

    if sample['tempfile'] == True:
        sample['tempfile'] = f
    else:
        f.close()

    return sample
