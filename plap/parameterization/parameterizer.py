from spafe.utils.preprocessing import SlidingWindow
from spafe.features.gfcc import gfcc
from plap.parameterization.fvector import FeatureVector
from plap.core.preprocessor import Preprocessor
from plap.parameterization.filterbank import Filterbank
import numpy as np
import soundfile as sf


class Parameterizer:
    
    def __init__(
        self,
        audio_path: str,
        preprocessor: Preprocessor
    ) -> None:     
        self.preprocessor = preprocessor
        self.signal, self.sample_rate, self.windowed_blocks, self.dft_blocks = preprocessor.preprocess(audio_path)

    @staticmethod
    def parameterize(audio_path: str, fvector: FeatureVector, preprocessor: Preprocessor = None):
        """
        """
        # Create default preprocessor
        if preprocessor is None:
            preprocessor = Preprocessor()

        # Create Parameterizer object
        parameterizer = Parameterizer(audio_path=audio_path, preprocessor=preprocessor)

        first = True
        for feature in fvector.features:
            if first:
                fvector.values = np.array([parameterizer.calc_feature(feature, fvector.features[feature])])
                first = False
            else:
                fvector.values = np.append(fvector.values, parameterizer.calc_feature(feature, fvector.features[feature]))

    def calc_feature(self, feature: str, feature_args: list) -> np.ndarray:
        num_ceps = feature_args[0] if feature_args is not None else 13
        spafe_window = SlidingWindow(self.preprocessor._block_size/self.sample_rate, self.preprocessor._step/self.sample_rate, "hanning" if self.preprocessor._window_type == "hann" else self.preprocessor._window_type)
        return {
            "rms": self.rms(),
            "zcr": self.zcr(),
            "SC": self.sc(),
            "gfcc": gfcc(sig=self.signal, fs=self.sample_rate, num_ceps=num_ceps, pre_emph=(self.preprocessor._preemphasis_coeff is not None), window=spafe_window, nfft=self.preprocessor._block_size),
            "mfcc": self.mfcc(feature_args)
        }[feature]
    
    def rms(self):
        return np.array([np.sqrt(np.mean(block**2)) for block in self.windowed_blocks])
    
    def zcr(self):
        return np.array([np.divide(np.sum(np.abs(np.subtract(np.sign(block[:-1]), np.sign(block[1:])))), 2) for block in self.windowed_blocks])
    
    def sc(self):
        ns = np.arange(self.preprocessor._block_size)
        fs = self.sample_rate / self.preprocessor._block_size * ns
        return np.array([np.divide(np.sum(np.multiply(fs, block)), np.sum(block) + (1e-10 if np.sum(block) == 0 else 0)) for block in np.abs(self.dft_blocks)])
    
    def mfcc(self, *args):
        if args[0] is not None:
            ncoeffs = args[0][0]
            nbands = args[0][1] if len(args[0]) > 1 else 6
        else:
            ncoeffs = 13
            nbands = 6
        block_size = self.preprocessor._block_size

        # Create mel filterbank
        mel_filterbank_params = [self.sample_rate, block_size, nbands]
        mel_filterbank = Filterbank(name="mel", params=mel_filterbank_params)

        # Filter each frame and sum the energy
        step = self.preprocessor._step
        nblocks = (self.signal.size - block_size) // step + 1
        x = np.zeros((nbands, nblocks))
        for b in range(0, nblocks):
            for i in range(0, nbands):
                acc = 0
                for k in range(0, block_size // 2 + 1):
                    acc += abs(self.dft_blocks[b][k]) * mel_filterbank[i][k]
                x[i][b] = acc

        # Apply log to each coefficient (mel filtered energy sum) for each frame
        xl = self.__apply_log(x)

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

    def __apply_log(self, arr: np.ndarray) -> np.ndarray:
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


parameterize = Parameterizer.parameterize