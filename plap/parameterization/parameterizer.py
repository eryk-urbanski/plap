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
        # self.signal, self.sample_rate, self.windowed_blocks, self.dft_blocks = preprocessor.preprocess(audio_path)
        self.signal, self.sample_rate = preprocessor.preprocess(audio_path)
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

        # fvector.values = np.empty([len(fvector.features)])
        # for index, (feature, _) in enumerate(fvector.features.items()):
        #     start_time = time.time()
        #     v = parameterizer.calc_feature(feature=feature, feature_args=fvector.features[feature])
        #     fvector.values[index] = v
        #     print(f"{feature} execution time: {time.time() - start_time}")



    def calc_feature(self, feature: str, feature_args: list) -> np.ndarray:
        feature = feature.lower()
        # Dictionary with deferred evaluation using lambda
        feature_map = {
            "lat": lambda: self.mpeg7.lat(),
            "tc": lambda: self.mpeg7.tc(),
            "sc": lambda: np.mean(self.mpeg7.sc()),
            "hsc": lambda: self.mpeg7.hsc(),
            "hsd": lambda: self.mpeg7.hsd(),
            "hss": lambda: self.mpeg7.hss(),
            "hsv": lambda: self.mpeg7.hsv(),
            "aff": lambda: np.mean(self.mpeg7.aff()),
            "asc": lambda: np.mean(self.mpeg7.asc()),
            "ass": lambda: np.mean(self.mpeg7.ass()),
            "asf": lambda: np.mean(self.mpeg7.asf()),
        }
        res = feature_map[feature]()
        return res
    

parameterize = Parameterizer.parameterize