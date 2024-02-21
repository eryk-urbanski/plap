import soundfile as sf


class AudioInfo:
    """
    A class used to load audio from files 
    and hold its properties and extracted features.

    ...

    Attributes
    ----------
    sample_rate : int
        The sample rate of audio
    signal : numpy.ndarray
        Audio data
    blocks: List[numpy.ndarray]
        Framed audio data
    windowed_blocks: List[numpy.ndarray]
        Frames multiplied by a window function
    dft_blocks: List[numpy.ndarray]
        Fast Fourier Transform (FFT) result for each frame

    """

    def __init__(self, file: str):
        """
        Parameters
        ----------
        file: str
            name of the audio file
        """

        self.signal, self.sample_rate = sf.read(file)
        self.blocks = []
        self.windowed_blocks = []
        self.dft_blocks = []
