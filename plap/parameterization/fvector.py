import numpy as np

class FeatureVector:
    """
    Provides an intuitive way of defining a set of features
    to be extracted from an audio signal.

    Supported features:
    Root Mean Square - "rms", no args
    Zero-Crossing Rate - "zcr", no args
    Spectral Centroid - "SC", no args
    Gammatone-Frequency Cepstral Coefficients - "gfcc", number of coefficients

    Examples:

    fvector = FeatureVector("rms", "mfcc", [13,], "SC")
    Here, rms is a feature without a need of additional parameters, 
    mfcc can take in parameters and SC again doesn't need any.

    fvector = FeatureVector("rms", [123, 34], "mfcc", [13,], "SC")
    This is incorrect (rms cannot accept any additional parameters) 
    and an error would be raised.

    fvector = FeatureVector("rms", "mfcc", "SC")
    This would work by setting the default additional parameters for mfcc. NOT YET
    """

    # SUPPORTED_CEPSTRAL_FEATURES = {
    #     "gfcc": [],
    #     "mfcc": [],
    #     }
    
    # SUPPORTED_TIMEDOMAIN_FEATURES = {
    #     "rms": None,
    #     "zcr": None,
    #     }
    
    SUPPORTED_MPEG7_FEATURES = {
        # Basic Spectral
        "ASF": None,
        "ASC": None,
        "ASS": None,
        # Signal Parameters
        "AFF": None,
        # Timbral Temporal
        "LAT": None,
        "TC": None,
        # Timbral Spectral
        "SC": None,
        "HSC": None,
    }

    SUPPORTED_FEATURES = {
        # **SUPPORTED_TIMEDOMAIN_FEATURES,
        # **SUPPORTED_CEPSTRAL_FEATURES,
        **SUPPORTED_MPEG7_FEATURES
        }

    def __init__(self, *args):
        """
        """
        self.features = {}
        self.parse_args(*args)
        self.values = 0

    def parse_args(self, *args):
        i = 0
        while i < len(args):
            feature = args[i]
            if not isinstance(feature, str) or feature not in self.SUPPORTED_FEATURES:
                raise ValueError(f"Invalid feature: {feature}")
            if self.SUPPORTED_FEATURES[feature] is not None:
                if i + 1 >= len(args) or not isinstance(args[i + 1], list):
                    raise ValueError(f"Missing or invalid parameters for '{feature}' feature")
                self.features[feature] = args[i + 1]
                i += 1
            else:
                self.features[feature] = None
            i += 1