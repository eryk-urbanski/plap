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
        Divides an audio signal into frames.
        Currently does not support overlapping.
        Adds or sets the AudioInfo object's blocks attribute.

        Parameters
        ----------
        audio_info : AudioInfo
            The input audio_info object.
        block_size : int
            The size of each frame.
        overlap : int
            The overlapping rate.

        """
        # step = round((100 - overlap) / 100.0 * block_size)
        step = block_size
        length = audio_info.signal.size
        nblocks = int(np.floor(length/step) + 1)
        setattr(audio_info, "blocks", np.zeros((nblocks, block_size)))
        for i in range(nblocks-1):
            audio_info.blocks[i] = audio_info.signal[i*step : i*step + block_size]
        if length % step != 0:
            remaining_samples = length % step
            last_block = np.pad(
                audio_info.signal[-remaining_samples:],
                (0, block_size - remaining_samples),
                mode="constant",
            )
            audio_info.blocks[-1] = last_block

    @staticmethod
    def windowing(audio_info: AudioInfo, window_type: str):
        """
        Applies a window function to each frame.
        Currently supports window types available in scipy's signal module.
        Adds or sets the AudioInfo object's windowed_blocks attribute.

        Parameters
        ----------
        audio_info : AudioInfo
            The input audio_info object.
        window_type : str
            The window type.

        """
        w = get_window(window=window_type, Nx=len(audio_info.blocks[0]))
        windowed_blocks = np.multiply(audio_info.blocks[:], w)
        setattr(audio_info, "windowed_blocks", windowed_blocks)

    @staticmethod
    def fft(audio_info: AudioInfo):
        """
        Compute the Fast Fourier Transform (FFT) for each frame.
        Adds or sets the AudioInfo object's dft_blocks attribute.

        Parameters
        ----------
        audio_info : AudioInfo
            The input audio_info object.

        """
        dft_blocks = np.apply_along_axis(scifft, 1, audio_info.windowed_blocks)
        setattr(audio_info, "dft_blocks", dft_blocks)
        
