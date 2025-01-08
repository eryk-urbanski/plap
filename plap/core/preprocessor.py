from typing import Tuple

import soundfile as sf
import numpy as np
from scipy.signal import lfilter
import librosa


class Preprocessor:
    """
    A class aggregating methods for signal pre-emphasis and fft.

    """

    def __init__(
        self,
        preemphasis_coeff: int = 0.68,
        block_size: int = 512,
        overlap: int = 50,
        window_type: str = "hann",
    ) -> None:
        """
        Preprocessor class instances are created to hold the various
        preprocessing parameters desired by the user.

        Parameters
        ----------
        block_size : int
            Size of each frame\n
                default value = 512
        overlap : int
            Overlapping rate\n
                default value = 50
        window_type : str
            Name of the window type\n
                default value = "hann"
        preemhpasis_coeff : int
            Pre-emphasis coefficient\n
            If set to None, pre-emphasis is not performed\n
                default value = 0.68

        """
        self._preemphasis_coeff = preemphasis_coeff
        self._block_size = block_size
        self._step = int((100 - overlap) / 100 * block_size)
        self._window_type = window_type

    def preprocess(self, audio_path: str) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Loads an audio signal from given path and performs preprocessing.

        Parameters
        ----------
        audio_path : str
            Path to the audio file

        Returns
        -------
        signal : numpy.ndarray
            Raw audio data
        sample_rate : int
            Sample rate (sampling frequency)
        stft_spectrum : numpy.ndarray
            Complex spectrum computed using librosa.stft

        """
        signal, sample_rate = sf.read(audio_path)
        if self._preemphasis_coeff is not None:
            signal = self.__preemphasis(signal)
        stft_spectrum = self.__stft(signal)
        return signal, sample_rate, stft_spectrum

    def __preemphasis(self, signal: np.ndarray) -> np.ndarray:
        """
        Applies pre-emphasis to the given signal.

        Parameters
        ----------
        signal : numpy.ndarray
            Signal to be pre-emphasised

        Returns
        -------
        signal : numpy.ndarray
            Pre-emphasised signal

        """
        return lfilter([1, -self._preemphasis_coeff], [1], signal)

    def __stft(self, signal):
        """
        Performs Short-Time Fourier Transform on the given signal.

        Parameters
        ----------
        signal : numpy.ndarray
            Signal to perform STFT on

        Returns
        -------
        stft : numpy.ndarray

        """
        stft = librosa.stft(
            y=signal,
            n_fft=self._block_size,
            hop_length=self._step,
            window=self._window_type
        )
        return stft