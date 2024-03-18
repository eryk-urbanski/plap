from plap.core.audio_info import AudioInfo
from plap.core.preprocessing import Preprocessing
from plap.parameterization.filterbank import Filterbank
import numpy as np


class Cepstral:
    """
    Provides a comprehensive pipeline for the extraction
    of selected cepstral coefficients.
    Currently supports MFCCs.

    """
    def __init__(
        self,
        audio_info: AudioInfo,
        block_size: int,
        window_type: str,
        overlap: int,
        filterbank_name: str,
        ncoeffs: int,
    ):
        """
        Cepstral class instances are created to hold the audio signal
        as well as various parameters for preprocessing, filter bank
        creation and coefficient extraction.

        """
        self.audio_info = audio_info
        self.block_size = block_size
        self.window_type = window_type
        self.overlap = overlap
        self.step = round((100 - overlap) / 100 * block_size)
        self.filterbank_name = filterbank_name
        self.ncoeffs = ncoeffs

    @staticmethod
    def mfcc(
        audio_info: AudioInfo, ncoeffs: int, nbands: int, block_size: int, window_type: str, overlap: int
    ) -> np.ndarray:
        """
        Calculates MFCCs for a given audio signal.

        Parameters
        ----------
        audio_info : AudioInfo
            The input audio_info object.
        ncoeffs : int
            Number of coefficients to be calculated.
        nbands : int
            Number of mel bands.
        block_size : int
            The size of each frame.
        window_type : str
            Name of the window type.
        overlap : int
            The overlapping rate.

        Returns
        -------
        Numpy array with the desired number of Mel-Frequency Cepstral Coefficients

        """
        # Idea: here create a Cepstral class instance called MFCCExtractor that will hold
        # preprocessing params, filterbank name, etc.
        # Preprocessing COMMON, GENERALLY
        # Get mel filter bank CHANGES
        # Filter each frame and sum the energy COMMON, GENERALLY
        # Apply log to each coefficient (filtered energy sum) for each frame COMMON, GENERALLY
        # Get desired num of coefficients for
        # each frame by applying dct to log mel filtered energy sums COMMON, GENERALLY
        pass

    def __preprocess(self) -> np.ndarray:
        """
        Performs necessary preprocessing on signal.

        Parameters
        ----------
        ?

        Returns
        -------
        ?

        """
        blocks = Preprocessing.framing(
            audio_info=self.audio_info,
            # block_size=self.preprocessing_params["Block Size"]
        )
        windowed_blocks = Preprocessing.windowing(
            blocks=blocks,
            window_type=self.window_type
        )
        dft_blocks = Preprocessing.fft(windowed_blocks=windowed_blocks)
        return dft_blocks

    def __create_filterbank(self, params: list) -> np.ndarray:
        """
        Creates a filter bank.

        Parameters
        ----------
        ?

        Returns
        -------
        ?

        """
        # Idea: takes in a list of params. self contains filter name, so an appropriate
        # function from filterbanks module is called and params is passed. Each of those functions
        # in that module knows how params is structured for them. For example, for mfcc params contains
        # sample_rate, block_size and nmel_bands (number of mel bands).
        return Filterbank(name=self.filterbank_name, params=params)

    def __apply_filterbank(self, filterbank) -> np.ndarray:
        # Applying each type of filterbanks can be different
        # so functions TODO in filterbank.py
        pass

    def __apply_log(self):
        pass

    def __apply_dct(self):
        pass
