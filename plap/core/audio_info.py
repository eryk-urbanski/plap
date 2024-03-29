import soundfile as sf


class AudioInfo:
    """
    Facilitates audio input from files 
    and holding its properties and extracted features.

    ...

    Attributes
    ----------
    sample_rate : int
        The sample rate of audio
    signal : numpy.ndarray
        Audio data

    """

    def __init__(self, file: str):
        """
        Parameters
        ----------
        file: str
            name of the audio file
        """
        self.signal, self.sample_rate = sf.read(file)