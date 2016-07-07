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
import os
import subprocess

from krotos.msd import Dataset
from krotos.msd.latent.features import LatentFeatures
from krotos.paths import ROOT_PATH
from krotos.debug import report, report_newline



d = Dataset.instance(new=True)
lf = LatentFeatures()
batch = d.minibatch(20, mapping='LATENT_FEATURES', trim=False, audio_tempfile=True)

song_labels = {}
echonest    = lf._echonest

unique_tracks_path = os.path.join(ROOT_PATH, 'msd/resources/unique_tracks.txt')
if not os.path.exists(unique_tracks_path): raise Exception("unique_tracks.txt not found.")
with open(unique_tracks_path, 'r') as unique_tracks:
    i = 0
    for line in unique_tracks:
        _, song_id, artist, track = line.strip().split("<SEP>")
        song_labels[song_id] = (artist + ' - ' + track)
        i += 1
        if (i % 5000 == 0):
            report("{0:7d} song labels...".format(i), sameline=True)

    report_newline()

sid_mismatches_path = os.path.join(ROOT_PATH, 'msd/resources/sid_mismatches.txt')
if not os.path.exists(sid_mismatches_path): raise Exception("sid_mismatches.txt not found.")
with open(sid_mismatches_path, 'r') as sid_mismatches:
    i = 0
    for line in sid_mismatches:
        song_labels[line[8:26]] = "<bad data: mismatched song>"
        i += 1
        if (i % 100 == 0):
            report("{0:5d} erroneous song labels noted...".format(i), sameline=True)

    report_newline()

for sample in batch:
    s           = sample['spectrogram_image']
    features    = sample['latent_features']
    title       = sample['title']
    artist_name = sample['artist_name']
    f           = sample['tempfile']

    closest = lf.closest(features, n=200, ordered=True)

    report('\t' + artist_name + ' - ' + title)
    report('\t' + '---')

    for track_id_echonest, distance, norm in closest:
        report('\t\t' + '{0:7.5f} ({1:7.5f})'.format(distance, norm) + '\t' + song_labels.get(track_id_echonest, ""))

    report_newline()


    p = subprocess.Popen(['vlc', f.name])

    plt.clf()
    librosa.display.specshow(s, hop_length=1024, x_axis='time', y_axis='mel')
    plt.colorbar(format='%1.3f')

    plt.title(artist_name.encode('utf-8', 'ignore') + ' - ' + title.encode('utf-8', 'ignore'))

    plt.show()

    p.kill()
    f.close()
