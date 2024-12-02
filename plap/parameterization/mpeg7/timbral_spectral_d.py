from typing import Tuple

import numpy as np
import librosa
from scipy.ndimage import uniform_filter1d

class TimbralSpectralD:
    ## List of Timbral Spectral Descriptors
    #------------------------------------------
    #  - Spectral Centroid Descriptor SC
    #  - Harmonic Spectral Centroid Descriptor HSC
    #  - Harmonic Spectral Deviation Descriptor HSD
    #  - Harmonic Spectral Spread Descriptor HSS
    #  - Harmonic Spectral Variation Descriptor HSV 
    #------------------------------------------

    # TODO document functions (translate descriptions from my thesis)

    def __init__(self, aw: np.ndarray, sample_rate: int, block_size: int, step: int, stft_magnitude: np.ndarray) -> None:

        self.aw = aw
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.step = step
        self.stft_magnitude = stft_magnitude

        # self.harmonic_spectrum, self.harmonic_freqs = self.__harmonic_peaks_estimation()
        self.harmonic_spectrum, self.harmonic_freqs = self.__harmonic_decomposition()

        self.sc_d = None
        self.hsc_d = None
        self.hsd_d = None
        self.hss_d = None
        self.hsv_d = None

    def sc(self):
        if self.sc_d is None:
            self.sc_d = self.__sc()
        return self.sc_d

    def hsc(self):
        if self.hsc_d is None:
            self.hsc_d, self.hsd_d, self.hss_d, self.hsv_d = self.__harmonic_descriptors()
        return self.hsc_d
    
    def hsd(self):
        if self.hsd_d is None:
            self.hsc_d, self.hsd_d, self.hss_d, self.hsv_d = self.__harmonic_descriptors()
        return self.hsd_d
    
    def hss(self):
        if self.hss_d is None:
            self.hsc_d, self.hsd_d, self.hss_d, self.hsv_d = self.__harmonic_descriptors()
        return self.hss_d
    
    def hsv(self):
        if self.hsv_d is None:
            self.hsc_d, self.hsd_d, self.hss_d, self.hsv_d = self.__harmonic_descriptors()
        return self.hsv_d
    

    def __sc(self):
        spectral_centroid = librosa.feature.spectral_centroid(
            S=self.stft_magnitude,
        )
        return spectral_centroid

    def __harmonic_decomposition(self):
        h, _ = librosa.decompose.hpss(self.stft_magnitude)
        freqs = librosa.fft_frequencies(
            sr=self.sample_rate,
            n_fft=self.block_size
        )
        return h, freqs
    
    def __harmonic_peaks_estimation(self):
        freqs = librosa.fft_frequencies(
            sr=self.sample_rate,
            n_fft=self.block_size
        )
        f0 = librosa.yin(
            y=self.aw, 
            sr=self.sample_rate, 
            fmin=62.5, 
            fmax=1500, 
            frame_length=self.block_size, 
            hop_length=self.step
        )
        harmonics = np.arange(1, 25)
        harmo_freqs = harmonics[:, np.newaxis] * f0
        f0_harm = librosa.f0_harmonics(self.stft_spectrum, freqs=freqs, f0=f0, harmonics=harmonics)
        return np.abs(f0_harm), harmo_freqs

    def __harmonic_descriptors(self):
        nb_frames = self.harmonic_spectrum.shape[1]

        # Instantaneous descriptors
        i_hsc = np.zeros(nb_frames)
        i_hsd = np.zeros(nb_frames)
        i_hss = np.zeros(nb_frames)
        i_hsv = np.zeros(nb_frames)

        i_nrg = np.zeros(nb_frames)
        old_ampl = np.zeros_like(self.harmonic_spectrum[:, 0])

        freqs = self.harmonic_freqs
        freq_len = len(freqs)
        eps = np.finfo(float).eps

        for i in range(nb_frames):
            ampl = self.harmonic_spectrum[:, i]
            dot_ampl_ampl = np.dot(ampl, ampl) + eps

            # inrg
            i_nrg[i] = np.sqrt(dot_ampl_ampl)

            # ihsc
            tmp = np.dot(freqs, ampl) / (np.sum(ampl, axis=0) + eps)
            i_hsc[i] = tmp

            # ihss
            num = np.dot((ampl * (freqs-tmp)), (ampl * (freqs-tmp))) # Faster than np.sum((ampl * (freqs-tmp))**2)
            i_hss[i] = 1/(tmp+eps) * np.sqrt(num/dot_ampl_ampl)

            # ihsd
            smoothed_spectral_env = uniform_filter1d(ampl, size=3, mode="nearest")
            tmp = np.sum(np.abs(np.log10(ampl+eps) - np.log10(smoothed_spectral_env+eps)))
            i_hsd[i] = tmp / freq_len

            # ihsv
            if i > 0:
                crossprod = np.dot(old_ampl, ampl) + eps  # Faster than np.sum(old_ampl*ampl)
                autoprod1 = np.dot(old_ampl, old_ampl) + eps # Faster than np.sum(old_ampl**2)
                autoprod2 = dot_ampl_ampl
                i_hsv[i] = 1 - crossprod / np.sqrt(autoprod1*autoprod2)

            old_ampl[:] = ampl
        
        threshold_positions = np.where(i_nrg > np.max(i_nrg) * 0.1)

        hsc = np.mean(i_hsc[threshold_positions])
        hsd = np.mean(i_hsd[threshold_positions])
        hss = np.mean(i_hss[threshold_positions])
        hsv = np.mean(i_hsv[threshold_positions])
        return hsc, hsd, hss, hsv
