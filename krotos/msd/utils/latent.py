from krotos.msd.latent.features import LatentFeatures



def get_latent_features(track_id_echonest):
    return LatentFeatures().get(track_id_echonest)[0]
