from typing import Tuple

import numpy as np
import librosa

class CepstralD:
    ## List of Supported Cepstral Descriptors
    #--------------------------------------------
    #  - Mel-Frequency Cepstral Coefficients MFSS
    #--------------------------------------------

    def __init__(self, aw: np.ndarray, sample_rate, block_size, step, window_type: str):

        self.aw = aw
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.step = step
        self.window_type = window_type

        self.mfcc_d = None

    def mfcc(self, num_coeffs: int = 20):
        if self.mfcc_d is None:
            self.mfcc_d = self.__mfcc(num_coeffs)
        return self.mfcc_d
    

    def __mfcc(self, num_coeffs: int):
        mfccs = librosa.feature.mfcc(
            y=self.aw,
            sr=self.sample_rate,
            n_mfcc=num_coeffs,
            n_fft=self.block_size,
            hop_length=self.step,
            window=self.window_type
        )
        mfccs = np.mean(mfccs, axis=1)
        return mfccs