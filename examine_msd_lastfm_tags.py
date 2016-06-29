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



d = Dataset.instance(new=True)
batch = d.minibatch(5, mapping='LASTFM_TAGS', trim=False, audio_tempfile=True)

for sample in batch:
    s           = sample['spectrogram_image']
    title       = sample['title']
    artist_name = sample['artist_name']
    tags        = sample['misc']['tag_names']
    f           = sample['file']

    p = subprocess.Popen(['vlc', f.name])

    plt.clf()
    librosa.display.specshow(s, x_axis='time', y_axis='mel')
    plt.colorbar(format='%1.3f')

    plt.title(artist_name.encode('utf-8', 'ignore') + ' - ' + title.encode('utf-8', 'ignore') + ' | ' + ', '.join(tags))

    plt.show()

    p.kill()
    f.close()