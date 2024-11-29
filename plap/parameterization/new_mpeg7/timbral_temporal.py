from typing import Tuple

import numpy as np
from scipy.signal import hilbert
import librosa

class TimbralTemporal:
    ## List of Timbral Temporal Descriptors
    #------------------------------------------
    #  - Log Attack Time Descriptor LAT
    #  - Temporal Centroid Descriptor TC
    #------------------------------------------

    def __init__(self, aw: np.ndarray, sample_rate: int, block_size: int, step: int) -> None:

        self.aw = aw
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.step = step

        self.lat_d = None
        self.tc_d = None

    def lat(self) -> float:
        if self.lat_d is None:
            self.lat_d, self.tc_d = self.__lat_tc()
        return self.lat_d
    
    def tc(self) -> float:
        if self.tc_d is None:
            self.lat_d, self.tc_d = self.__lat_tc()
        return self.tc_d
    

    def __lat_tc(self) -> Tuple[float, float]:
        threshold_percent = 2

        amplitude_envelope = librosa.feature.rms(y=self.aw, frame_length=self.block_size, hop_length=self.step)[0]
        frames = range(len(amplitude_envelope))
        t = librosa.frames_to_time(frames, sr=self.sample_rate, hop_length=self.step)

        temporal_centroid = np.divide(
            np.sum(np.multiply(amplitude_envelope, t)), np.sum(amplitude_envelope)
        )

        stop_attack_value, stop_attack_pos = max(amplitude_envelope), np.argmax(amplitude_envelope)
        threshold = stop_attack_value * threshold_percent / 100
        start_attack_pos = np.where(amplitude_envelope > threshold)[0][0]
        if start_attack_pos == stop_attack_pos:
            start_attack_pos -= 1

        log_attack_time = np.log10(t[stop_attack_pos] - t[start_attack_pos])
        print(f"t[stop], t[start]: {t[stop_attack_pos], t[start_attack_pos]}")

        return log_attack_time, temporal_centroid