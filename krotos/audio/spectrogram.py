import numpy as np
import librosa



# TODO: Add separate methods for loading spectrograms from the dataset and
#   miscellaneous samples

def mel_spectrogram(audio_path):
    # TODO: Investigate whether log scaling the power spectrum should be done
    # TODO: Consider min-max normalization

    y, sr = librosa.load(audio_path,
        sr = 22050,
        duration=29.0,
        mono=True
    )
    s = librosa.feature.melspectrogram(y, sr, n_mels=128, hop_length=512)

    return s
