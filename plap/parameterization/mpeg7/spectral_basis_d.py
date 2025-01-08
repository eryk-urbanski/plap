from typing import Tuple

import numpy as np
from sklearn.decomposition import FastICA

class SpectralBasisD:
    ## List of Spectral Basis Descriptors
    #------------------------------------------
    #  - Audio Spectrum Basis Descriptor ASB
    #  - Audio Spectrum Projection Descriptor ASP
    #------------------------------------------

    # TODO document functions (translate descriptions from my thesis)

    def __init__(self, ase: np.ndarray):

        self.ase = ase
        # self.sample_rate = sample_rate
        # self.block_size = block_size
        # self.window_type = window_type
        # self.magnitude = stft_magnitude

        self.asb_d = None
        self.asp_d = None

    # Audio Spectrum Basis ASB
    def asb(self, num_ic: int = 20):
        if self.asb_d is None:
            self.asb_d, self.asp_d = self.__asb_asp(num_ic)
        return self.asb_d
    
    # Audio Spectrum Basis ASB
    def asp(self):
        if self.asp_d is None:
            self.asb_d, self.asp_d = self.__asb_asp(20)
        return self.asp_d
    

    def __asb_asp(self, num_ic):
        X = self.ase
        eps = np.finfo(float).eps
        # Convert power spectrum to dB scale
        X = 10*np.log10(X + eps)
        # Extract L2-norm of total spectral energy envelope
        env = np.sqrt(np.sum(X.T * X.T, axis=1))
        # Extract spectral shape
        X = X.T / (env[:, np.newaxis])
        # Extract spectral shape basis functions
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        # Basis Function Dimension Reduction
        V = Vt.T[:, :num_ic]

        # optional: ICA
        Y = X @ V
        ica = FastICA(n_components=num_ic, whiten="arbitrary-variance", algorithm='deflation')
        V2 = ica.fit_transform(Y.T)
        V = V @ np.linalg.pinv(V2).T

        # Project spectral shape onto spectral basis functions
        P = np.hstack((env[:, np.newaxis], X @ V))

        return np.mean(V, axis=0), np.mean(P, axis=0)
