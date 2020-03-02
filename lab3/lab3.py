import librosa
import numpy as np
import librosa.onset


def norm(x):
    return(x - x[0])/x.mean()


normed = np.vectorize(norm)


def get_time_envelope(x, sr, hop_length=256):
    onset_envelope = librosa.onset.onset_strength(x, sr=sr,
                                                  hop_length=hop_length)
    N = len(x)
    T = N/float(sr)
    t = np.linspace(0, T, len(onset_envelope))
    return t, onset_envelope/np.mean(onset_envelope)


def get_seq(arr, seq_len):
    data = np.empty((len(arr)-seq_len, seq_len))
    for i in range(len(arr)-seq_len):
        chunk = np.empty(seq_len)
        for j in range(seq_len):
            chunk[j] = arr[i + j]
        data[i] = chunk
    return data
