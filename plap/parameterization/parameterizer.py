from plap.parameterization.fvector import FeatureVector
from plap.core.preprocessor import Preprocessor
from plap.parameterization.mpeg7 import MPEG7
import numpy as np


class Parameterizer:
    
    def __init__(
        self,
        audio_path: str,
        preprocessor: Preprocessor
    ) -> None:     
        self.preprocessor = preprocessor
        self.signal, self.sample_rate, self.windowed_blocks, self.dft_blocks = preprocessor.preprocess(audio_path)
        self.mpeg7 = MPEG7(self.signal, self.sample_rate)

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
        return {
            "LAT": self.mpeg7.lat(),
            "TC": self.mpeg7.tc(),
            "SC": np.mean(self.mpeg7.sc()),
            "HSC": self.mpeg7.hsc(),
            "HSS": self.mpeg7.hss(),
            "HSV": self.mpeg7.hsv(),
            "AFF": np.mean(self.mpeg7.aff()),
            "ASC": np.mean(self.mpeg7.asc()),
            "ASS": np.mean(self.mpeg7.ass()),
            "ASF": np.mean(self.mpeg7.asf()),
        }[feature]
    

parameterize = Parameterizer.parameterize