from plap.core.audio_info import AudioInfo
from scipy.fft import fft as scifft
from scipy.signal.windows import get_window
import numpy as np


class Preprocessing:
    """
    A class aggregating methods for signal framing with/without overlapping, windowing and fft

    """

    @staticmethod
    def framing(audio_info: AudioInfo, block_size: int, overlap: int):
        """
        Divide an audio signal into overlapping frames.

        Parameters
        ----------
        audio_info : AudioInfo
            The input audio_info object.
        block_size : int
            The size of each frame.
        overlap : int
            The overlapping rate.

        """
        step = round((100 - overlap) / 100 * block_size)
        audio_info.blocks = []
        length = audio_info.signal.size
        for i in range(0, length - block_size, step):
            audio_info.blocks.append(audio_info.signal[i : i + block_size])
        # Performs zero-padding on the last block if necessary
        if length % block_size != 0:
            remaining_samples = length % block_size
            last_block = np.pad(
                audio_info.signal[-remaining_samples:],
                (0, block_size - remaining_samples),
                mode="constant",
            )
            audio_info.blocks.append(last_block)

    @staticmethod
    def windowing(audio_info: AudioInfo, window_type: str):
        """
        Apply a window function to each frame.
        Currently supports window types available in scipy's signal module.

        Parameters
        ----------
        audio_info : AudioInfo
            The input audio_info object.
        window_type : str
            The window type.

        """
        w = get_window(window=window_type, Nx=len(audio_info.blocks[0]))
        for block in audio_info.blocks:
            audio_info.windowed_blocks.append(block * w)

    @staticmethod
    def fft(audio_info: AudioInfo):
        """
        Compute the Fast Fourier Transform (FFT) for each frame.

        Parameters
        ----------
        audio_info : AudioInfo
            The input audio_info object.

        """
        for block in audio_info.windowed_blocks:
            audio_info.dft_blocks.append(scifft(block))
