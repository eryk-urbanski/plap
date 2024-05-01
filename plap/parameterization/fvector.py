class FeatureVector:
    """
    Provides an intuitive way of defining a set of features
    to be extracted from an audio signal.

    Supported features:


    Examples:

    fvector = FeatureVector("rms", "mfcc", [13,], "SC")
    Here, rms is a feature without a need of additional parameters, 
    mfcc can take in parameters and SC again doesn't need any.

    fvector = FeatureVector("rms", [123, 34], "mfcc", [13,], "SC")
    This is be incorrect (rms cannot accept any additional parameters) 
    and an error would be raised.

    fvector = FeatureVector("rms", "mfcc", "SC")
    This would work by setting the default additional parameters for mfcc.
    """

    def __init__(self, *args):
        """
        """
        self.features = {"rms": None, "mfcc": []}
        self.parse_args(*args)

    def parse_args(self, *args):
        i = 0
        while i < len(args):
            feature = args[i]
            if feature not in self.features:
                raise ValueError(f"Invalid feature: {feature}")
            if self.features[feature] is not None:
                if i + 1 >= len(args) or not isinstance(args[i + 1], list):
                    raise ValueError(f"Missing or invalid parameters for '{feature}' feature")
                self.features[feature] = args[i + 1]
                i += 1
            i += 1

