import numpy as np
import librosa



def mel_spectrogram(audio_path, duration=29.0):
    try:
        y, sr = librosa.load(audio_path,
            sr = 22050,
            duration=duration,
            mono=True
        )
    except EOFError as e:
        return False, None

    # Require audio samples to meet duration minimum.
    if librosa.get_duration(y=y, sr=sr) < duration: return False, None

    s   = librosa.feature.melspectrogram(y, sr, n_mels=128, hop_length=1024)

    # Log scale
    s   = librosa.logamplitude(s, ref_power=np.max, top_db=80.0)

    # Min-max scale to [0.0, 1.0]
    s  += 80.0
    s  *= 0.0125 # multiplication is faster than division

    return True, s.T
