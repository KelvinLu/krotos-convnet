from multiprocessing.dummy import Pool as ThreadPool
import tempfile
import time

from krotos.exceptions import ParametersError
from krotos.msd.utils import lastfm, msd_hdf5, sevendigital
from krotos.msd.latent.features import LatentFeatures
from krotos.audio import spectrogram
from krotos.debug import report, report_newline



WORKERS = 4


# kwarg mapping may be one of the following:
#   'LATENT_FEATURES'
#   'LASTFM_TAGS'
def make_minibatch(dataset, n=10, mapping='LATENT_FEATURES', trim=False, audio_tempfile=False):
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

    report("Minibatch: {} samples downloaded and processed in {}s ({}s process time).".format(n, time.time() - time_start_world, time.clock() - time_start_proc), sameline=False)

    if trim:
        results = [(sample['spectrogram_image'], sample['mapping']) for sample in results]

    return results

def output_latent_features_mapping(track_id_echonest):
    latents = LatentFeatures().get(track_id_echonest)[0]

    if latents is None: return None, None

    return latents, None

def output_lastfm_tags_mapping(track_id):
    tag_vector, tag_names, num_tags = lastfm.get_tag_data(track_id)

    if not num_tags: return None, None

    return tag_vector, {
        'tag_names':    tag_names,
        'num_tags':     num_tags,
    }

def select_samples(dataset, n, mapping, audio_tempfile=False):
    samples = []

    # Get metadata and Last.fm tags for a track.
    # Do sqlite database accesses single-threaded.
    while len(samples) < n:
        sample_ind          = dataset._sample_training_ind()
        track_id, metadata  = msd_hdf5.get_summary([sample_ind])

        track_id                                                    = track_id[0]
        track_id_7digital, track_id_echonest, title, artist_name    = metadata[0]

        if not track_id_7digital: continue

        mapping_output, misc = None, None

        if mapping == 'LATENT_FEATURES':
            mapping_output, misc = output_latent_features_mapping(track_id_echonest)
        elif mapping == 'LASTFM_TAGS':
            mapping_output, misc = output_lastfm_tags_mapping(track_id)
        else:
            raise ParametersError('Bad value for parameter mapping')

        if mapping_output is None: continue

        samples.append({
            'track_id':             track_id,
            'track_id_7digital':    track_id_7digital,
            'track_id_echonest':    track_id_echonest,
            'title':                title,
            'artist_name':          artist_name,
            'mapping':              mapping_output,
            'misc':                 misc,
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
