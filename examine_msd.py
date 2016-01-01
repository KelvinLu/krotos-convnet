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



d = Dataset()
batch = d.human_examine(5)

for s, tags, mp3, title, artist_name in batch:
    p = subprocess.Popen(['vlc', mp3.name])

    plt.clf()
    librosa.display.specshow(librosa.logamplitude(s, ref_power=np.max), x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')

    plt.title(artist_name.encode('utf-8') + ' - ' + title.encode('utf-8') + ' | ' + ', '.join(tags))

    plt.show()

    p.kill()
    mp3.close()
