from typing import Tuple

import numpy as np
import librosa

class BasicSpectralD:
    ## List of Basic Spectral Descriptors
    #------------------------------------------
    #  - Audio Spectrum Envelope Descriptor ASE
    #  - Audio Spectrum Centroid Descriptor ASC
    #  - Audio Spectrum Spread Descriptor   ASS
    #  - Audio Spectrum Flatness Descriptor ASF
    #------------------------------------------

    # TODO document functions (translate descriptions from my thesis)

    def __init__(self, aw: np.ndarray, sample_rate: int, block_size: int, window_type: str, stft_magnitude: np.ndarray):

        self.aw = aw
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.window_type = window_type
        self.magnitude = stft_magnitude
        self.powers = self.__get_powers(self.magnitude)

        self.ase_d = None
        self.asc_d = None
        self.ass_d = None
        self.asf_d = None
        self.asf_variance_across_bands_d = None

    def ase(self):
        """
        Calculate the Audio Spectrum Envelope Descriptor (ASE)

        """
        if self.ase_d is None:
            self.ase_d = self.__ase()
        return self.ase_d

    # Audio Spectrum Centroid ASC
    def asc(self):
        if self.asc_d is None:
            self.asc_d, self.ass_d = self.__asc_ass()
        return self.asc_d

    # Audio Spectrum Spread ASS
    def ass(self):
        if self.ass_d is None:
            self.asc_d, self.ass_d = self.__asc_ass()
        return self.ass_d

    # Audio Spectrum Flatness ASF
    def asf(self, variance_across_bands: bool = False):
        if self.asf_d is None:
            self.asf_d, self.asf_variance_across_bands_d = self.__asf()
        if variance_across_bands is True:
            return self.asf_variance_across_bands_d
        else:
            return self.asf_d


    def __get_powers(self, magnitude: np.ndarray):
        # TODO get it from BasicD as rms calculated from stft_magnitude, but it's meaned through freqs, 
        # so would need modifications in __asc_ass()
        powers = magnitude**2
        powers[1:-1, :] = 2 * powers[1:-1, :]
        return powers

    def __ase(self):
        return self.powers

    def __asc_ass(self) -> Tuple[np.ndarray, np.ndarray]:
        low_freq = 62.5
        fft_freq = librosa.fft_frequencies(
            sr=self.sample_rate,
            n_fft=self.block_size
        )

        # Replace frequencies less than 62.5Hz with nominal freq 31.25Hz
        num_less_low_freq = np.sum(fft_freq < low_freq)
        fft_freq = np.concatenate(([31.25], fft_freq[num_less_low_freq:]))

        # Log-scaled frequencies relative to 1kHz
        fft_freq_log = np.log2(fft_freq / 1000)

        powers = self.powers
        # Sum powers for the frequencies below loedge
        if num_less_low_freq > 1:
            summed_powers = np.sum(powers[:num_less_low_freq, :], axis=0, keepdims=True)
            powers = np.concatenate((summed_powers, powers[num_less_low_freq:, :]), axis=0)

        # Calculate the Audio Spectrum Centroid
        audio_spectrum_centroid = np.sum(fft_freq_log[:, np.newaxis] * powers, axis=0) / (np.sum(powers, axis=0) + np.finfo(float).eps)

        # Calculate the Audio Spectrum Spread
        numframes = powers.shape[1]
        a = np.outer(fft_freq_log, np.ones(numframes)) - np.outer(np.ones(len(fft_freq_log)), audio_spectrum_centroid)
        audio_spectrum_spread = np.sqrt(np.sum((a**2) * powers, axis=0) / (np.sum(powers, axis=0) + np.finfo(float).eps))
        return audio_spectrum_centroid, audio_spectrum_spread

    def __asf(self) -> np.ndarray:

        low_edge = 250
        high_edge = 16000

        fftout = librosa.stft(
            y=self.aw,
            n_fft=self.block_size,
            hop_length=self.block_size,
            window=self.window_type,
        )
        fftout = np.abs(fftout) ** 2
        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.block_size)

        num_bands = int(np.floor(4 * np.log2(high_edge / low_edge)))

        audio_spectrum_flatness = np.zeros((num_bands, fftout.shape[1]))

        for k in range(num_bands):
            # Get the frequency indices for the current band
            f_low = low_edge * (2 ** (k / 4))
            f_high = high_edge * (2 ** ((k+1) / 4))
            band_start = np.searchsorted(freqs, f_low)
            band_end = np.searchsorted(freqs, f_high)

            # Extract the power spectrum for the band
            band_spectrum = fftout[band_start:band_end, :]
            audio_spectrum_flatness[k, :] = librosa.feature.spectral_flatness(S=band_spectrum)

        audio_spectrum_flatness_v = np.var(audio_spectrum_flatness.T, axis=0)
        audio_spectrum_flatness = np.mean(audio_spectrum_flatness.T, axis=0)
        return audio_spectrum_flatness, audio_spectrum_flatness_v