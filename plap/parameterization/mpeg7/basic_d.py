from typing import Tuple

import numpy as np
import librosa

class BasicD:
    ## List of Basic Descriptors
    #------------------------------------------
    #  - Audio Waveform Descriptor AW
    #  - Audio Power Descriptor AP
    #       - Implemented as more useful RMS
    #------------------------------------------

    def __init__(self, signal: np.ndarray, stft_magnitude: np.ndarray, sample_rate: int, block_size: int, step: int) -> None:

        self.stft_magnitude = stft_magnitude
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.step = step

        self.aw_d = signal
        self.ap_d = self.__ap()

    def aw(self):
        return self.aw_d
    
    def ap(self):
        return self.ap_d
    

    def __ap(self):
        ap = librosa.feature.rms(S=self.stft_magnitude, frame_length=self.block_size, hop_length=self.step)[0]
        return ap