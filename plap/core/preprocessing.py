from plap.core.audio_info import AudioInfo
from scipy.fft import fft as scifft
from scipy.signal.windows import get_window


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
        for i in range(0, audio_info.signal.size - block_size, step):
            audio_info.blocks.append(audio_info.signal[i : i + block_size])

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
