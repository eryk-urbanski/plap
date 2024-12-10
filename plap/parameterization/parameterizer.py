from plap.parameterization.fvector import FeatureVector
from plap.core.preprocessor import Preprocessor
from plap.parameterization.mpeg7.basic_d import BasicD
from plap.parameterization.mpeg7.basic_spectral_d import BasicSpectralD
from plap.parameterization.mpeg7.signal_parameters import SignalParameters
from plap.parameterization.mpeg7.timbral_temporal_d import TimbralTemporalD
from plap.parameterization.mpeg7.timbral_spectral_d import TimbralSpectralD
from plap.parameterization.mpeg7.spectral_basis_d import SpectralBasisD
from plap.parameterization.cepstral.cepstral_d import CepstralD
import numpy as np


class Parameterizer:
    
    def __init__(
        self,
        audio_path: str,
        preprocessor: Preprocessor
    ) -> None:     
        self.preprocessor = preprocessor if preprocessor is not None else Preprocessor()
        self.signal, self.sample_rate, self.stft_spectrum = preprocessor.preprocess(audio_path)
        self.magnitude = np.abs(self.stft_spectrum)

        self._basic_parameterizer = BasicD(
            signal=self.signal,
            sample_rate=self.sample_rate,
            block_size=self.preprocessor._block_size,
            step=self.preprocessor._step,
            stft_magnitude=self.magnitude
        )
        self._basic_spectral_parameterizer = BasicSpectralD(
            aw=self.signal,
            sample_rate=self.sample_rate,
            block_size=self.preprocessor._block_size,
            window_type=self.preprocessor._window_type,
            stft_magnitude=self.magnitude
        )
        self._timbral_temporal_parameterizer = TimbralTemporalD(
            ap=self._basic_parameterizer.ap(),
            sample_rate=self.sample_rate,
            step=self.preprocessor._step
        )
        self._timbral_spectral_parameterizer = TimbralSpectralD(
            aw=self.signal,
            sample_rate=self.sample_rate,
            block_size=self.preprocessor._block_size,
            step=self.preprocessor._step,
            stft_magnitude=self.magnitude
        )
        self._signal_parameters_parameterizer = SignalParameters(
            aw=self.signal,
            sample_rate=self.sample_rate,
            block_size=self.preprocessor._block_size,
            step=self.preprocessor._step
        )
        self._spectral_basis_parameterizer = SpectralBasisD(
            ase=self._basic_spectral_parameterizer.ase()
        )
        self._cepstral_parameterizer = CepstralD(
            aw=self.signal,
            sample_rate=self.sample_rate,
            block_size=self.preprocessor._block_size,
            step=self.preprocessor._block_size,
            window_type=self.preprocessor._window_type,
        )

    @staticmethod
    def parameterize(audio_path: str, fvector: FeatureVector, preprocessor: Preprocessor = None):
        """
        """
        # Create Parameterizer object
        parameterizer = Parameterizer(audio_path=audio_path, preprocessor=preprocessor)

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
            "sc": lambda: np.mean(self._timbral_spectral_parameterizer.sc()),
            "sc_var": lambda: np.var(self._timbral_spectral_parameterizer.sc()),
            "hsc": lambda: self._timbral_spectral_parameterizer.hsc(),
            "hsd": lambda: self._timbral_spectral_parameterizer.hsd(),
            "hss": lambda: self._timbral_spectral_parameterizer.hss(),
            "hsv": lambda: self._timbral_spectral_parameterizer.hsv(),
            "aff": lambda: np.mean(self._signal_parameters_parameterizer.aff()),
            "aff_var": lambda: np.var(self._signal_parameters_parameterizer.aff()),
            "ase": lambda: np.mean(self._basic_spectral_parameterizer.ase(), axis=0),
            "asc": lambda: np.mean(self._basic_spectral_parameterizer.asc()),
            "asc_var": lambda: np.var(self._basic_spectral_parameterizer.asc()),
            "ass": lambda: np.mean(self._basic_spectral_parameterizer.ass()),
            "ass_var": lambda: np.var(self._basic_spectral_parameterizer.ass()),
            "asf": lambda: self._basic_spectral_parameterizer.asf(),
            "asf_mean": lambda: np.mean(self._basic_spectral_parameterizer.asf()),
            "asf_var": lambda: self._basic_spectral_parameterizer.asf(variance_across_bands=True),
            "asf_var_mean": lambda: np.mean(self._basic_spectral_parameterizer.asf(variance_across_bands=True)),
            "asb": lambda: self._spectral_basis_parameterizer.asb(),
            "asb_mean": lambda: np.mean(self._spectral_basis_parameterizer.asb()),
            "asp": lambda: self._spectral_basis_parameterizer.asp(),
            "asp_mean": lambda: np.mean(self._spectral_basis_parameterizer.asp()),

            "mfcc": lambda: self._cepstral_parameterizer.mfcc(),
        }
        res = feature_map[feature]()
        return res
    

parameterize = Parameterizer.parameterize