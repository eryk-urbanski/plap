from scipy.signal.windows import get_window
from scipy.signal import lfilter
import numpy as np
import soundfile as sf
from typing import Tuple


class Preprocessor:
    """
    A class aggregating methods for signal framing, windowing and fft.

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
            Size of each frame
                default value = 512
        overlap : int
            Overlapping rate
                default value = 50
        window_type : str
            Name of the window type
                default value = "hann"
        preemhpasis_coeff : int
            Pre-emphasis coefficient.
            If set to None, pre-emphasis is not performed
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
        sample_rate : int
            Sample rate (sampling frequency)
        windowed_blocks : numpy.ndarray
            Windowed signal frames
            Shape: (nblocks, block_size)
        dft_blocks : numpy.ndarray
            FFT blocks (specra)
            Shape: (nblocks, block_size // 2 + 1)

        """
        signal, sample_rate = sf.read(audio_path)
        if self._preemphasis_coeff is not None:
            signal = self.__preemphasis(signal)
        blocks = self.__framing(signal)
        windowed_blocks = self.__windowing(blocks)
        dft_blocks = self.__fft(windowed_blocks)
        return sample_rate, windowed_blocks, dft_blocks

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

    def __framing(self, signal: np.ndarray) -> np.ndarray:
        """
        Divides an audio signal into overlapping frames.

        Parameters
        ----------
        signal : numpy.ndarray
            Signal to be framed

        Returns
        -------
        blocks : numpy.ndarray
            Framed signal
            Shape: (nblocks, block_size)

        """
        length = signal.size
        nblocks = (length - self._block_size + self._step) // self._step
        nremaining_samples = (length - self._block_size + self._step) % self._step
        if nremaining_samples != 0:
            nblocks = nblocks + 1
        nblocks = int(nblocks)
        blocks = np.zeros((nblocks, self._block_size))
        for i in range(nblocks - 1):
            blocks[i] = signal[i * self._step : i * self._step + self._block_size]
        if nremaining_samples != 0:
            last_block = np.pad(
                signal[-nremaining_samples:],
                (0, self._block_size - nremaining_samples),
                mode="constant",
            )
            blocks[-1] = last_block
        return blocks

    def __windowing(self, blocks: np.ndarray) -> np.ndarray:
        """
        Applies a window function to each frame.
        Currently supports window types available in scipy's signal module.

        Parameters
        ----------
        blocks: numpy.ndarray
            Framed signal

        Returns
        -------
        windowed_blocks : numpy.ndarray
            Windowed signal frames
            Shape: (nblocks, block_size)

        """
        w = get_window(window=self._window_type, Nx=self._block_size)
        windowed_blocks = np.multiply(blocks[:], w)
        return windowed_blocks

    def __fft(self, windowed_blocks: np.ndarray) -> np.ndarray:
        """
        Computes the Fast Fourier Transform (FFT) for each frame.

        Parameters
        ----------
        windowed_blocks : numpy.ndarray
            Windowed signal frames
            Shape: (nblocks, block_size)

        Returns
        -------
        dft_blocks : numpy.ndarray
            FFT blocks
            Shape: (nblocks, block_size // 2 + 1)

        """
        dft_blocks = np.apply_along_axis(np.fft.rfft, 1, windowed_blocks)
        return dft_blocks
