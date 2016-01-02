import numpy as np
import librosa



def mel_spectrogram(audio_path):
    y, sr = librosa.load(audio_path,
        sr = 22050,
        duration=29.0,
        mono=True
    )

    # Require audio samples to meet duration minimum.
    if librosa.get_duration(y=y, sr=sr) < 29.0: return False, None

    s   = librosa.feature.melspectrogram(y, sr, n_mels=128, hop_length=512)

    # Log scale
    s   = librosa.logamplitude(s, ref_power=np.max, top_db=80.0)

    # Min-max scale to [0.0, 1.0]
    s  += 80.0
    s  *= 0.0125 # multiplication is faster than division

    return True, s
