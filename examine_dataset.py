# Examine the MagnaTagATune dataset. Running this script will take a handful of
# MagnaTagATune audio samples and for each...
#
# ... show its Mel spectrogram in a matplotlib figure
# ... launch VLC to play the tune
#
# To progress between samples, just close the spectrogram figure

import matplotlib.pyplot as plt
import librosa
import numpy as np
import subprocess

from krotos.magnatagatune import Dataset



d = Dataset()
batch = d.human_examine(10)

for s, tags, mp3_path in batch:
    p = subprocess.Popen(['vlc', mp3_path])

    plt.clf()
    librosa.display.specshow(librosa.logamplitude(s, ref_power=np.max), x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')

    plt.title(', '.join(tags) or 'No tags')

    plt.show()

    p.kill()
