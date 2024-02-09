import soundfile as sf

class AudioInfo:
    """
    A class used to load audio form files 
    and hold its properties and extracted features

    ...

    Attributes
    ----------
    sample_rate : int
        The sample rate of audio
    signal : numpy.ndarray
        Audio data
    blocks: numpy.ndarray list
        Framed audio data

    Methods
    -------
    
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
