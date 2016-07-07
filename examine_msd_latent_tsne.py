# Examine the Million Song Dataset. Running this script will visualize each Echo
# Nest Taste Profile songs' latent features with t-SNE.

import os, subprocess
import matplotlib.pyplot as plt
from matplotlib.text import Annotation
import numpy as np
import scipy
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering

from krotos.msd.latent.features import LatentFeatures
from krotos.paths import ROOT_PATH
from krotos.debug import report, report_newline



report("Get LatentFeatures instance...")
lf          = LatentFeatures()



report("Scaling latent features")
latents     = lf.Y
# latents     = StandardScaler().fit_transform(latents)



samples     = 5000
report("Selecting {0} random samples...".format(samples))
sample_idxs = np.random.choice(latents.shape[0], size=samples, replace=False)
latents     = latents[sample_idxs, :]



report("Performing t-SNE embedding...")
distance    = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(latents, metric='cosine'))
model       = TSNE(n_components=2, method='barnes_hut', metric='precomputed')
embedding   = model.fit_transform(distance)



report("Performing clustering...")
n_clusters      = 10
db              = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage='average',
                    affinity='cosine'
                ).fit(latents)
cluster_labels  = db.labels_
cmap = plt.get_cmap('jet', n_clusters)


report("Getting norms...")
norms           = np.linalg.norm(latents, axis=1)
# min_max_norms   = (norms - np.min(norms)) / (np.max(norms) - np.min(norms))



report("Retrieving song labels...")
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



report("Plotting...")
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(embedding[:, 0], embedding[:, 1], c=cluster_labels, cmap=cmap, s=100, alpha=0.5, linewidths=0.0, picker=True)

last_annotation = None
last_latent     = None
last_ind        = None

def report_top(idx, n):
    report("\tGlobal closest songs:")
    closest = lf.closest(lf.Y[idx], n=n, ordered=True)
    i = 0
    for echonest_id, score, norm in closest:
        i += 1
        report("\t\t{0}\t{1:7.5} ({2:7.5}): {3}".format(i, score, norm, song_labels[echonest_id]))

def onpick(event):
    global last_annotation
    global last_latent
    global last_ind

    ind = event.ind[0]
    idx = sample_idxs[ind]
    x, y = embedding[ind]
    track_id_echonest = echonest.get_track_id(idx)
    label = unicode(song_labels[track_id_echonest], errors='ignore')

    if last_ind == ind:
        report_top(idx, 100)
        return
    last_ind = ind

    if last_latent is not None:
        s = np.dot(last_latent, latents[ind]) / (np.linalg.norm(last_latent) * np.linalg.norm(latents[ind]))
        d = np.linalg.norm(last_latent - latents[ind])
        report('Cosine similarity: {0: 7.5f}'.format(s))
        report('L2 distance:       {0: .5f}'.format(d))
    last_latent = latents[ind]

    report('Selected point at (x={1: 7.3f}, y={2: 7.3f}, norm={3: 7.3f}): \t{0}.'.format(label, x, y, norms[ind]))
    if last_annotation is not None: last_annotation.remove()
    last_annotation = Annotation(
        label,
        xy=(x, y),
        xytext=(-20, 20),
        textcoords='offset points',
        ha='right',
        va='bottom',
        bbox=dict(boxstyle = 'round,pad=0.5', fc='white', alpha=0.6),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', alpha=0.4)
    )

    ax.add_artist(last_annotation)
    fig.canvas.draw()

fig.canvas.mpl_connect('pick_event', onpick)



report("Displaying plot")
plt.show()
report_newline()
