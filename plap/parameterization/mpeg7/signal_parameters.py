from typing import Tuple

import numpy as np
import librosa
from scipy.ndimage import uniform_filter1d

class SignalParameters:
    ## List of Signal Parameters
    #------------------------------------------
    #  - Audio Fundamental Frequency Descriptor AFF
    #  - Audio Harmonicity Descriptor AH
    #------------------------------------------

    # TODO document functions (translate descriptions from my thesis)

    def __init__(self, aw: np.ndarray, sample_rate: int, block_size: int, step: int, low_edge: float = 62.5, high_edge: float = 1500) -> None:

        self.aw = aw
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.step = step
        self.low_edge = low_edge
        self.high_edge = high_edge

        self.aff_d = None
        self.ah_d = None

    def aff(self):
        """
        Audio Fundamental Frequency AFF

        """
        if self.aff_d is None:
            self.aff_d = self.__aff()
        return self.aff_d
    
    def ah(self):
        """
        Audio Harmonicity AH

        """
        if self.ah_d is None:
            self.ah_d = self.__ah()
        return self.ah_d
    

    def __aff(self):
        f0_yin = librosa.yin(
            y=self.aw, 
            sr=self.sample_rate,
            fmin=self.low_edge,
            fmax=self.high_edge,
            frame_length=self.block_size,
            hop_length=self.step
        )
        return f0_yin
    
    def __ah(self):
        # TODO
        pass