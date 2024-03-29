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
        self._audio_info = audio_info
        self._block_size = block_size
        self._window_type = window_type
        self._overlap = overlap
        self._step = round((100 - overlap) / 100 * block_size)
        self._filterbank_name = filterbank_name
        self._ncoeffs = ncoeffs

    @staticmethod
    def mfcc(
        audio_info: AudioInfo,
        ncoeffs: int,
        nbands: int,
        block_size: int,
        window_type: str,
        overlap: int,
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
        mfccs : numpy.ndarray
            Desired number of Mel-Frequency Cepstral Coefficients
            Shape: (ncoeffs, nblocks)

        """
        MFCCExtractor = Cepstral(
            audio_info=audio_info,
            block_size=block_size,
            window_type=window_type,
            overlap=overlap,
            filterbank_name="mel",
            ncoeffs=ncoeffs,
        )
        # Perform necessary preprocessing
        dft_blocks = MFCCExtractor.__preprocess()

        # Create mel filterbank
        mel_filterbank_params = [audio_info.sample_rate, block_size, nbands]
        mel_filterbank = MFCCExtractor.__create_filterbank(params=mel_filterbank_params)

        # Filter each frame and sum the energy
        step = MFCCExtractor._step
        nblocks = (audio_info.signal.size - block_size) // step + 1
        x = np.zeros((nbands, nblocks))
        for b in range(0, nblocks):
            for i in range(0, nbands):
                acc = 0
                for k in range(0, block_size // 2 + 1):
                    acc += abs(dft_blocks[b][k]) * mel_filterbank[i][k]
                x[i][b] = acc

        # Apply log to each coefficient (mel filtered energy sum) for each frame
        xl = MFCCExtractor.__apply_log(x)

        # Get desired num of mfcc coefficients for each frame
        # by applying dct to log mel filtered energy sums
        mfccs = np.zeros((ncoeffs, nblocks))
        for b in range(0, nblocks):
            for j in range(0, ncoeffs):
                acc = 0
                for i in range(0, nbands):
                    acc += xl[i][b] * np.cos(j * (i - 0.5) * np.pi / nbands)
                mfccs[j][b] = acc
        return mfccs

    def __preprocess(self) -> np.ndarray:
        """
        Performs necessary preprocessing on signal.
        Currently supported cepstral features share the same
        preprocessing pipeline composed of framing, windowing and fft.

        Returns
        -------
        dft_blocks : numpy.ndarray
            Preprocessed signal.
            Shape: (nblocks, block_size)

        """
        dft_blocks = Preprocessing.preprocess(
            audio_info=self._audio_info,
            framing=True,
            windowing=True,
            fft=True,
            params=[
                self._block_size,
                self._overlap,
                self._window_type])
        return dft_blocks

    def __create_filterbank(self, params: list) -> np.ndarray:
        """
        Creates a filter bank.

        Parameters
        ----------
        params : list
            Parameters needed for the creation of the desired filterbank.
            Mel: sample_rate, block_size, nmel_bands
            Gammatone: TODO

        Returns
        -------
        Filterbank object.

        """
        # Idea: takes in a list of params. self contains filter name, so an appropriate
        # function from filterbanks module is called and params is passed. Each of those functions
        # in that module knows how params is structured for them. For example, for mfcc params contains
        # sample_rate, block_size and nmel_bands (number of mel bands).
        return Filterbank(name=self._filterbank_name, params=params)

    @staticmethod
    def __apply_log(arr: np.ndarray) -> np.ndarray:
        # Handle zeros entering log function
        """
        Helper function for handling zeros entering log function.

        Parameters
        ----------
        arr : numpy.ndarray

        Returns
        -------
        The same array without any zero values.

        """
        arr = np.where(arr == 0, arr + 1e-9, arr)
        return np.log10(arr)
