import librosa
import numpy as np
import noisereduce as nr
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter1d

def _stft_parameters(sample_rate):
    
    num_freq = 1025
    frame_shift_ms = 12.5
    frame_length_ms = 50
    
    n_fft = (num_freq - 1) * 2
    hop_length = int(frame_shift_ms / 1000 * sample_rate)
    win_length = int(frame_length_ms / 1000 * sample_rate)
    
    return n_fft, hop_length, win_length

def load_audio(audio_file, sample_rate):
    y, sr = librosa.load(audio_file, sr=sample_rate)
    
    if y.shape[0]< 960000:
        new_y= np.zeros(960000, dtype= np.float32)
        new_y[:y.shape[0]] =y
        return new_y , sr
    else:
        return y, sr

def get_spectrogram(y,sr= 16000, num_mels= 80, apply_denoise=False,return_audio=False):    
    if apply_denoise:
        y= denoise(y, sr)
    
    n_fft, hop_length, win_length = _stft_parameters(sr)
    
    feat = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=num_mels,
                                                  n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    feat = np.log(feat + 1e-6) # log-scaled

    feat = [feat]
    feat = np.concatenate(feat, axis=0)
    feat = (feat - feat.mean(axis=1)[:,np.newaxis]) / (feat.std(axis=1)+1e-16)[:,np.newaxis]
    
    if return_audio:
        return feat, y
    else:
        return feat

def _envelope(y, rate, threshold):
    mask = []
    y_mean = maximum_filter1d(np.abs(y), mode="constant", size=rate//20)
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask, y_mean
    
def denoise(y, sr, threshold=0.25):
    mask, env = _envelope(y, sr, threshold)
    y_denoise = nr.reduce_noise(audio_clip=y, noise_clip=y[np.logical_not(mask)], verbose=False)
    return y_denoise

def plot_feature(feat):
    plt.figure(figsize=(18, 3))
    plt.imshow(feat, aspect="auto", origin="lower", cmap='magma')
    plt.colorbar()
    plt.tight_layout()