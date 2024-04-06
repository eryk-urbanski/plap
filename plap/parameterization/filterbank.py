import numpy as np
from scipy.signal.windows import triang

class Filterbank:
    """
    Provides ... TODO

    """

    def __new__(self, name: str, params: list):
        return {
            "mel": self.__mel_filterbank(params=params),
            "gammatone": self.__gammatone_filterbank(params=params),
        }[name]

    @staticmethod
    def __mel_filterbank(params: list) -> np.ndarray:
        """
        smth TODO

        Parameters
        ----------
        params : list
            sample_rate, block_size, nmel_bands

        """
        # asserts TODO
        sample_rate = params[0]
        block_size = params[1]
        nmel_bands = params[2]

        # Convert the highest frequency to mel
        max_freq_hz = sample_rate / 2
        max_freq_mel = 2595 * np.log10(1 + max_freq_hz / 700)

        # Create mel_bands equally spaced points (centres of mel bands)
        # mel_centres includes both the centre points as well
        # as the lowest and highest frequency
        mel_centres = np.linspace(0, max_freq_mel, nmel_bands + 2)

        # Convert these points back to Hz
        hz_centres = np.round(700 * (10 ** (mel_centres / 2595) - 1))

        # Find indices of the nearest frequency bins
        freqs = np.linspace(0, sample_rate / 2, block_size // 2 + 1)
        hz_centres_indices = np.zeros_like(hz_centres, dtype=int)
        for i, hz_val in enumerate(hz_centres):
            closest = np.argmin(np.abs(freqs - hz_val))
            hz_centres_indices[i] = closest

        # Create mel filter bank
        mel_filterbank = np.zeros((nmel_bands, block_size // 2 + 1))
        for i in range(0, nmel_bands):
            low = hz_centres_indices[i]
            high = hz_centres_indices[i + 2]
            mel_filterbank[i][low:high] = triang(high - low)

        return mel_filterbank


    @staticmethod
    def __gammatone_filterbank(sample_rate, num_filters, f_min, order=4):


        f_max = sample_rate / 2

        # Generating center frequencies for the filters using ERB formula
        def erb(f):
            return 21.4 * np.log10(0.00437 * f + 1)
        # formula by Glasberg & Moore
        erb_min = erb(f_min)
        erb_max = erb(f_max)
        erb_values = np.linspace(erb_min, erb_max, num_filters)
        cf = (10 ** (erb_values / 21.4) - 1) / 0.00437

        filterbank = np.zeros((num_filters, int(sample_rate / 2 + 1)))

        for i in range(num_filters):
            f_center = cf[i]
            # Calculate bandwidth using ERB formula
            b = 1.019 * erb(f_center)

            # Creating filter impulse
            t = np.arange(0, int(sample_rate / 2 + 1)) / sample_rate
            impulse_response = t ** (order - 1) * np.exp(-2 * np.pi * b * t) * np.cos(2 * np.pi * f_center * t)

            # Applying the filter for each frequency
            filter_response = np.abs(np.fft.rfft(impulse_response))
            filterbank[i, :] = filter_response / np.max(filter_response)  # Normalization

        return filterbank