from plap.core.audio_info import AudioInfo
from plap.core.preprocessing import Preprocessing
import numpy as np
from scipy.signal.windows import triang
from scipy.fft import dct


class MFCC:
    """
    Provides a comprehensive pipeline for the extraction
    of Mel-frequency Cepstral Coefficients (MFCCs).

    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def mfcc(audio_info: AudioInfo, ncoeffs, mel_bands, block_size, overlap):
        """
        Calculates MFCCs for a given audio signal.

        Parameters
        ----------
        audio_info : AudioInfo
            The input audio_info object.

        """
        # Perform necessary preprocessing
        blocks = Preprocessing.framing(audio_info=audio_info, block_size=block_size, overlap=overlap)
        windowed_blocks = Preprocessing.windowing(blocks=blocks, window_type="hann")
        dft_blocks = Preprocessing.fft(windowed_blocks=windowed_blocks)

        # Filter each frame with mel filters and sum the energy
        step = round((100 - overlap) / 100.0 * block_size)
        nblocks = (audio_info.signal.size - block_size) // step + 1
        x = np.zeros((mel_bands, nblocks))
        mel_fbank = MFCC.mel_filterbank(audio_info.sample_rate, mel_bands, block_size)
        for b in range(0, nblocks):
            for i in range(0, mel_bands):
                acc = 0
                for k in range(0, block_size//2+1):
                    acc += abs(dft_blocks[b][k]) * mel_fbank[i][k]
                x[i][b] = acc

        # Apply log to each coefficient (mel filtered energy sum) for each frame
        x = np.where(x == 0, x + 1e-9, x)
        xl = np.log10(x)
                
        # Get desired num of mfcc coefficients for each frame by applying dct to log mel filtered energy sums
        mfccs = np.zeros((ncoeffs, nblocks))
        for b in range(0, nblocks):
            for j in range(0, ncoeffs):
                acc = 0
                for i in range(0, mel_bands):
                    acc += xl[i][b] * np.cos(j*(i-0.5)*np.pi/mel_bands)
                mfccs[j][b] = acc
        return mfccs

    @staticmethod
    def mel_filterbank(sample_rate, mel_bands, block_size):
        # Convert the highest frequency to mel
        max_freq_hz = sample_rate / 2
        max_freq_mel = MFCC.hz_to_mel(max_freq_hz)

        # Create mel_bands equally spaced points (centres of mel bands)
        # mel_centres includes both the centre points as well
        # as the lowest and highest frequency
        mel_centres = np.linspace(0, max_freq_mel, mel_bands + 2)

        # Convert these points back to Hz
        hz_centres = np.round(MFCC.mel_to_hz(mel_centres))

        # Find indices of the nearest frequency bins
        freqs = np.linspace(0, sample_rate / 2, block_size//2 + 1)
        hz_centres_indices = np.zeros_like(hz_centres, dtype=int)
        for i, hz_val in enumerate(hz_centres):
            closest = np.argmin(np.abs(freqs - hz_val))
            hz_centres_indices[i] = closest

        # Create mel filter bank
        mel_filterbank = np.zeros((mel_bands, block_size//2 + 1))
        for i in range(0, mel_bands):
            low = hz_centres_indices[i]
            high = hz_centres_indices[i+2]
            mel_filterbank[i][low:high] = triang(high-low)

        return mel_filterbank

    @staticmethod
    def hz_to_mel(freq_hz):
        return 2595 * np.log10(1 + freq_hz/700)

    @staticmethod
    def mel_to_hz(freq_mel):
        return 700 * (10**(freq_mel/2595) - 1)
