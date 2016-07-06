# Examine the Million Song Dataset. Running this script will take a handful of
# Million Song Dataset track samples and for each...
#
# ... show its Mel spectrogram in a matplotlib figure
# ... launch VLC to play the tune
#
# To progress between samples, just close the spectrogram figure

import matplotlib.pyplot as plt
import librosa
import numpy as np
import subprocess

from krotos.msd import Dataset
from krotos.msd.latent.features import LatentFeatures



d = Dataset.instance(new=True)
lf = LatentFeatures()
batch = d.minibatch(5, mapping='LATENT_FEATURES', trim=False, audio_tempfile=True)

for sample in batch:
    s           = sample['spectrogram_image']
    features    = sample['mapping']
    title       = sample['title']
    artist_name = sample['artist_name']
    f           = sample['tempfile']

    closest = lf.closest(features, n=20, ordered=True)

    print '\t' + artist_name + ' - ' + title
    print '\t' + '---'

    for track_id_echonest, distance in closest:
        print '\t\t' + str(track_id_echonest) + '\t' + str(distance)

    print '\t' + '==='


    p = subprocess.Popen(['vlc', f.name])

    plt.clf()
    librosa.display.specshow(s, hop_length=1024, x_axis='time', y_axis='mel')
    plt.colorbar(format='%1.3f')

    plt.title(artist_name.encode('utf-8', 'ignore') + ' - ' + title.encode('utf-8', 'ignore'))

    plt.show()

    p.kill()
    f.close()
