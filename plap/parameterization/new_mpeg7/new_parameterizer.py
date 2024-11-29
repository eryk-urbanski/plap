from plap.parameterization.fvector import FeatureVector
from plap.core.preprocessor import Preprocessor
from plap.parameterization.new_mpeg7.basic_spectral import BasicSpectral
from plap.parameterization.new_mpeg7.timbral_temporal import TimbralTemporal
import numpy as np


class NewParameterizer:
    
    def __init__(
        self,
        audio_path: str,
        preprocessor: Preprocessor
    ) -> None:     
        self.preprocessor = preprocessor if preprocessor is not None else Preprocessor()
        self.signal, self.sample_rate, self.magnitude, _ = preprocessor.preprocess(audio_path)

        self._basic_spectral_parameterizer = BasicSpectral(
            aw=self.signal,
            sample_rate=self.sample_rate,
            block_size=self.preprocessor._block_size,
            stft_magnitude=self.magnitude
            )
        self._timbral_temporal_parameterizer = TimbralTemporal(
            aw=self.signal,
            sample_rate=self.sample_rate,
            block_size=self.preprocessor._block_size,
            step=self.preprocessor._step
        )

    @staticmethod
    def parameterize(audio_path: str, fvector: FeatureVector, preprocessor: Preprocessor = None):
        """
        """
        # Create Parameterizer object
        parameterizer = NewParameterizer(audio_path=audio_path, preprocessor=preprocessor)

        first = True
        for feature in fvector.features:
            if first:
                fvector.values = np.array([parameterizer.calc_feature(feature)])
                first = False
            else:
                fvector.values = np.append(fvector.values, parameterizer.calc_feature(feature))

    def calc_feature(self, feature: str) -> np.ndarray:
        feature = feature.lower()
        # Dictionary with deferred evaluation using lambda
        feature_map = {
            "lat": lambda: self._timbral_temporal_parameterizer.lat(),
            "tc": lambda: self._timbral_temporal_parameterizer.tc(),
            # "sc": lambda: np.mean(self.mpeg7.sc()),
            # "hsc": lambda: self.mpeg7.hsc(),
            # "hsd": lambda: self.mpeg7.hsd(),
            # "hss": lambda: self.mpeg7.hss(),
            # "hsv": lambda: self.mpeg7.hsv(),
            # "aff": lambda: np.mean(self.mpeg7.aff()),
            "ase": lambda: np.mean(self._basic_spectral_parameterizer.ase()),
            "asc": lambda: np.mean(self._basic_spectral_parameterizer.asc()),
            "ass": lambda: np.mean(self._basic_spectral_parameterizer.ass()),
            "asf": lambda: np.mean(self._basic_spectral_parameterizer.asf()),
        }
        res = feature_map[feature]()
        return res
    

new_parameterize = NewParameterizer.parameterize