from audiomentations import Compose, AddGaussianNoise, AddGaussianSNR , PitchShift , Gain
import colorednoise as cn
import numpy as np

class AudioTransform:
    def __init__(self, always_apply=False, p=0.5):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, y: np.ndarray, sr: int):
        if self.always_apply:
            return self.apply(y)
        else:
            if np.random.rand() < self.p:
                return self.apply(y)
            else:
                return y

    def apply(self, y: np.ndarray):
        raise NotImplementedError
        
        
        
class PinkNoiseSNR(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5.0, max_snr=20.0, **kwargs):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        pink_noise = cn.powerlaw_psd_gaussian(1, len(y))
        a_pink = np.sqrt(pink_noise ** 2).max()
        augmented = (y + pink_noise * 1 / a_pink * a_noise).astype(y.dtype)
        return augmented
    

AUGMENT = Compose([PitchShift(min_semitones=-4, max_semitones=4, p=0.2),
                   Gain(min_gain_in_db=-12, max_gain_in_db=12, p=0.2),
                   AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.2),
                   AddGaussianSNR(min_SNR=0.1, max_SNR=1, p=0.2),
                   PinkNoiseSNR(min_snr=5.0, max_snr=10, p=0.2)
                   ])

def do_aug(samples, sample_rate):
    return AUGMENT(samples, sample_rate)
