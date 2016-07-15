import numpy as np

from krotos.msd.latent.features import LatentFeatures



def get_latent_features(track_id_echonest, normalize=False):
    lf = LatentFeatures().get(track_id_echonest)[0]
    if lf is None: return None
    if normalize:
        norm = np.linalg.norm(lf)
        if norm != 0: lf = lf / norm
    return lf
