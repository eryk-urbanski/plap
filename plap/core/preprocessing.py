from plap.core.audio_info import AudioInfo
from scipy.fft import fft as scifft
from scipy.signal.windows import get_window
import numpy as np


class Preprocessing:
    """
    A class aggregating methods for signal framing, windowing and fft.

    """

    @staticmethod
    def framing(audio_info: AudioInfo, block_size: int, overlap: int) -> np.ndarray:
        """
        Divides an audio signal into frames.
        Currently does not support overlapping.

        Parameters
        ----------
        audio_info : AudioInfo
            The input audio_info object.
        block_size : int
            The size of each frame.
        overlap : int
            The overlapping rate.

        Returns
        -------
        blocks : numpy.ndarray
            Framed signal.
            Shape: (nblocks, block_size)

        """
        # step = round((100 - overlap) / 100.0 * block_size)
        step = block_size
        length = audio_info.signal.size
        nblocks = length // step + 1
        blocks = np.zeros((nblocks, block_size))
        for i in range(nblocks - 1):
            blocks[i] = audio_info.signal[i * step : i * step + block_size]
        if length % step != 0:
            remaining_samples = length % step
            last_block = np.pad(
                audio_info.signal[-remaining_samples:],
                (0, block_size - remaining_samples),
                mode="constant",
            )
            blocks[-1] = last_block
        return blocks

    @staticmethod
    def windowing(blocks: np.ndarray, window_type: str) -> np.ndarray:
        """
        Applies a window function to each frame.
        Currently supports window types available in scipy's signal module.

        Parameters
        ----------
        audio_info : AudioInfo
            The input audio_info object.
        window_type : str
            The window type.

        Returns
        -------
        windowed_blocks : numpy.ndarray
            Windowed signal frames.
            Shape: (nblocks, block_size)

        """
        w = get_window(window=window_type, Nx=len(blocks[0]))
        windowed_blocks = np.multiply(blocks[:], w)
        return windowed_blocks

    @staticmethod
    def fft(windowed_blocks: np.ndarray) -> np.ndarray:
        """
        Compute the Fast Fourier Transform (FFT) for each frame.

        Parameters
        ----------
        audio_info : AudioInfo
            The input audio_info object.

        Returns
        -------
        dft_blocks : numpy.ndarray
            FFT blocks.
            Shape: (nblocks, block_size)

        """
        dft_blocks = np.apply_along_axis(scifft, 1, windowed_blocks)
        return dft_blocks