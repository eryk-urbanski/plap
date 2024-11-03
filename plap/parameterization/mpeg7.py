from typing import Tuple
import math

import soundfile as sf
from scipy.signal.windows import hamming
import numpy as np
import librosa


class MPEG7:
    ## List of descriptors
    #------------------------------------------
    # Basic
    #  - Audio Waveform Descriptor AW TODO PORT
    #  - Audio Power Descriptor AP TODO PORT
    # Basic Spectral
    #  - Audio Spectrum Envelope Descriptor ASE TODO PORT
    #  - Audio Spectrum Centroid Descriptor ASC DONE PORT TODO COMPARISON
    #  - Audio Spectrum Spread Descriptor ASS DONE PORT TODO COMPARISON
    #  - Audio Spectrum Flatness Descriptor ASF DONE LIBROSA TODO COMPARISON
    # Signal Parameters
    #  - Audio Fundamental Frequency Descriptor AFF DONE PORT TODO COMPARISON
    #  - Audio Harmonicity Descriptor AH TODO PORT
    # Timbral Temporal
    #  - Log Attack Time Descriptor LAT DONE PORT TODO COMPARISON
    #  - Temporal Centroid Descriptor TC DONE PORT TODO COMPARISON
    # Timbral Spectral
    #  - Spectral Centroid Descriptor SC DONE LIBROSA TODO COMPARISON
    #  - Harmonic Spectral Centroid Descriptor HSC TODO PORT
    #  - Harmonic Spectral Deviation Descriptor HSD TODO PORT
    #  - Harmonic Spectral Spread Descriptor HSS TODO PORT
    #  - Harmonic Spectral Variation Descriptor HSV TODO PORT
    # Spectral Basic
    #  - Audio Spectrum Basis Descriptor ASB TODO PORT
    #  - Audio Spectrum Projection Descriptor ASP TODO PORT
    # Silence Descriptor SD
    #------------------------------------------


    config = {
        "n_fft": 1024,
        "hop_length": 512,
        "cutoff_freq": 20,
        "low_freq": 62.5,
        "high_freq": 1500,
    }
    
    def __init__(self, audio_data: np.ndarray, sample_rate: int):
        self.audio_data = audio_data
        self.sample_rate = sample_rate

        standvar = self._mpeg7init()
        self.hop_size = standvar["hop_size"]
        self.window_size = standvar["window_size"]
        self.fft_size = standvar["fft_size"]
        self.window = standvar["window"]

        # Additional fields needed for Timbral Spectral descriptors
        self.time_f0 = None
        self.f0_bp = None
        self.pos_f0 = None
        self.mf0_hz = None
        self.nbt0 = 8
        self.overlap_factor = 2
        self.L_sec = None
        self.X_m = None # STFT

    # ------------------
    # | Basic Spectral |
    # ------------------

    # Audio Spectrum Flatness ASF
    def asf(self) -> np.ndarray:
        """
        uses librosa spectral_flatness
        """
        spectral_flatness = librosa.feature.spectral_flatness(
            y=self.audio_data, n_fft=self.config["n_fft"], hop_length=self.config["hop_length"]
        )
        return spectral_flatness
    
    # Audio Spectrum Centroid ASC
    def asc(self) -> np.ndarray:
        audio_spectrum_centroid, _, _ = self._asc()
        return audio_spectrum_centroid

    # Audio Spectrum Spread ASS
    def ass(self) -> np.ndarray:
        audio_spectrum_centroid, fft_freq_log, powers = self._asc()
        numframes = powers.shape[1]

        # Create the matrix 'a' by subtracting centroid from each log frequency
        a = np.outer(fft_freq_log, np.ones(numframes)) - np.outer(np.ones(len(fft_freq_log)), audio_spectrum_centroid)
        # print(a.shape)
        # print("a[:10, 77:79]: " + str(a[:10, 77:79])) # !!! it gets very off after some rows/columns, weird
        # Calculate the Audio Spectrum Spread
        audio_spectrum_spread = np.sqrt(np.sum((a**2) * powers, axis=0) / (np.sum(powers, axis=0) + np.finfo(float).eps))
        return audio_spectrum_spread
    

    # ---------------------
    # | Signal Parameters |
    # ---------------------

    def aff(self) -> np.ndarray:

        Km = int(np.ceil(self.sample_rate / self.config['low_freq']))  # maximum lag
        Kl = self.sample_rate // self.config['high_freq']   # minimum lag
        hop_size = self.hop_size
        num_frames = (len(self.audio_data) - 2 * hop_size) // hop_size
        ProbFrame = int(np.ceil(Km / hop_size))

        f0 = np.zeros(num_frames)

        K = hop_size

        n = self.window_size # window size
        for frame in range(1, num_frames - ProbFrame + 1):
            m = (frame+1) * hop_size
            den1 = np.sum(self.audio_data[m:m + n] ** 2)
            phi = np.zeros(K)
            den = np.sum(self.audio_data[m - Kl:m - Kl + n] ** 2)

            for k in range(Kl, K + 1):
            # Normalized Cross-Correlation
                den = den - self.audio_data[m - k + n - 1] ** 2 + self.audio_data[m - k] ** 2
                num = np.sum(self.audio_data[m:m + n] * self.audio_data[m - k:m - k + n])
                phi[k - Kl] = num / (np.sqrt(den1 * den) + np.finfo(float).eps)

            mag = np.max(phi)
            index = np.argmax(phi > 0.97 * mag) + Kl

            f0[frame] = self.sample_rate / index

            if frame < ProbFrame:
                K = K + hop_size
            elif frame == ProbFrame:
                K = Km

        # Use second frame fundamental as the 1st frame estimate
        f0[0] = f0[1]

        # For the last ProbFrame-1 frames use the last estimate of fundamental frequency
        m = len(self.audio_data) - n
        den1 = np.sum(self.audio_data[m:m + n] ** 2)
        for k in range(Kl, K + 1):
            den = np.sum(self.audio_data[m - k:m - k + n] ** 2)
            num = np.sum(self.audio_data[m:m + n] * self.audio_data[m - k:m - k + n])
            phi[k - Kl] = num / (np.sqrt(den1 * den) + np.finfo(float).eps)

        mag = np.max(phi)
        index = np.argmax(phi) + Kl
        self.time_f0 = (np.arange(1, num_frames + 1) * self.hop_size) / self.sample_rate
        f0[frame + 1:num_frames] = self.sample_rate / index
        self.f0_bp = np.column_stack((self.time_f0, f0))
        self.pos_f0 = np.where(self.f0_bp[:, 1] > 10)[0]
        self.mf0_hz = np.median(self.f0_bp[self.pos_f0, 1])

        return f0

    # --------------------
    # | Timbral Temporal |
    # --------------------

    # Log Attack Time LAT
    def lat(
        self,
        cutoff_freq: int = config["cutoff_freq"],
        ds_factor: int = 3,
        threshold_percent: int = 2,
    ) -> np.float64:
        energy_bp = self._energy_breakpoint(
            audio_data=self.audio_data,
            sample_rate=self.sample_rate,
            cutoff_freq=cutoff_freq,
            ds_factor=ds_factor,
        )
        time_v = energy_bp[:, 0]
        energy_v = energy_bp[:, 1]

        stopattack_value, stopattack_pos = max(energy_v), np.argmax(energy_v)
        threshold = stopattack_value * threshold_percent / 100
        tmp = np.where(energy_v > threshold)[0]
        startattack_pos = tmp[0]
        if startattack_pos == stopattack_pos:
            startattack_pos -= 1

        log_attack_time = np.log10(time_v[stopattack_pos] - time_v[startattack_pos])
        return log_attack_time

    # Temporal Centroid TC
    def tc(
        self, cutoff_freq: int = config["cutoff_freq"], ds_factor: int = 3
    ) -> np.float64:
        energy_bp = self._energy_breakpoint(
            audio_data=self.audio_data,
            sample_rate=self.sample_rate,
            cutoff_freq=cutoff_freq,
            ds_factor=ds_factor,
        )
        time_v = energy_bp[:, 0]
        energy_v = energy_bp[:, 1]

        temporal_centroid = np.divide(
            np.sum(np.multiply(energy_v, time_v)), np.sum(energy_v)
        )
        return temporal_centroid


    # --------------------
    # | Timbral Spectral |
    # --------------------

    # Spectral Centroid SC
    def sc(self):
        """
        uses librosa spectral_centroid
        """
        spectral_centroid = librosa.feature.spectral_centroid(
            y=self.audio_data, sr=self.sample_rate, n_fft=self.config["n_fft"], hop_length=self.config["hop_length"]
        )
        return spectral_centroid
    
    # Harmonic Spectral Centroid HSC
    def hsc(self):
        harmonic_spectral_centroid = 0
        if self.X_m is None:
            self._h_spectre()
        X_m = self.X_m
        return harmonic_spectral_centroid
    

    # ------------------
    # | Helper methods |
    # ------------------

    def _mpeg7init(self):
        '''
        Creates a structure of the default values to be used throughout the descriptors.
        '''
        hopsize = [self.sample_rate * 30, 1000] # default 30ms 
        # Get the quotient, remainder numerator, and remainder denominator
        num = hopsize[0]
        den = hopsize[1]
        quot = num // den
        fact = math.gcd(num, den)
        remd = den // fact
        remn = num // fact
        remn = remn % remd
        q = quot
        n = remn
        d = remd
        if n == 0:
            hopsize = q
        else:
            print("INIT MPEG7: Unhandled case")

        windowsize = int(np.ceil(self.sample_rate * 30 / 1000))
        FFTsize = 2**int(np.ceil(np.log2(windowsize)))
        window = hamming(windowsize)

        standvar = {
            "hop_size": hopsize,
            "window_size": windowsize,
            "window": window,
            "fft_size": FFTsize,
        }
        return standvar

    def _asc(self):
        (fftout, phase) = self._getspec(
            audio_data=self.audio_data,
            sample_rate=self.sample_rate,
            hopsize=self.hop_size,
            n_fft=self.fft_size,
            window_size=self.window_size,
            window_type="hamming",
        )
        low_freq = 62.5
        num_frames = fftout.shape[1]
        # print("num_frames: ", num_frames)

        # Frequency vector (half of the spectrum)
        fft_freq = np.arange(self.fft_size // 2 + 1) * self.sample_rate / self.fft_size
        # print("fft_freq[400:405]: ", fft_freq[400:405])

        # Replace frequencies less than 62.5Hz with nominal freq 31.25Hz
        num_less_low_freq = np.sum(fft_freq < low_freq)
        fft_freq = np.concatenate(([31.25], fft_freq[num_less_low_freq:]))
        # print(fft_freq.shape)
        # print("fft_freq[400:405]: ", fft_freq[400:405])

        # Log-scaled frequencies relative to 1kHz
        fft_freq_log = np.log2(fft_freq / 1000)
        # print("fft_freq_log[400:405]: ", fft_freq_log[400:405])
        # print("fft_freq_log[50:55]: ", fft_freq_log[50:55])

        # Calculate powers !!! something off, check again after implementing specgram2 in getspec instead of librosa
        powers = fftout**2
        powers[1:-1, :] = 2 * powers[1:-1, :]
        # print("powers[400:405, 95:97]: ", powers[400:405, 95:97])
        # print("powers[:5, :2]: ", powers[:5, :2])
        # print(powers.shape)

        # Sum powers for the frequencies below loedge
        if num_less_low_freq > 1:
            summed_powers = np.sum(powers[:num_less_low_freq, :], axis=0, keepdims=True)
            powers = np.concatenate((summed_powers, powers[num_less_low_freq:, :]), axis=0)
        # print("powers[400:405, 95:97]: ", powers[400:405, 95:97])
        # print("powers[:5, :2]: ", powers[:5, :2])
        # print(powers.shape)

        # Calculate the Audio Spectrum Centroid !!! may be something wrong here as well
        audio_spectrum_centroid = np.sum(fft_freq_log[:, np.newaxis] * powers, axis=0) / (np.sum(powers, axis=0) + np.finfo(float).eps)
        return audio_spectrum_centroid, fft_freq_log, powers

    def _energy_breakpoint(
        self, audio_data: np.ndarray, sample_rate: int, cutoff_freq: int, ds_factor: int
    ) -> np.ndarray:
        """
        Calculates the energy breakpoint function.
        // Still not the exact results compared to matlab

        Parameters
        ----------
        audio_data: numpy.ndarray
            Numpy array containing data samples
        sample_rate: int
            Sampling rate in Hz
        cutoff_freq: int
            Cutoff frequency for low-pass filtering of the energy
        ds_factor: int
            Down-sampling factor for the energy

        Returns
        -------
        energy_bp: numpy.ndarray
            A 2D array with time [seconds] in the first column
            and energy values in the second column.

        """

        # Calculate the length of the filter window, ensuring it's odd
        filter_length = round(sample_rate / (2 * cutoff_freq))
        filter_length += filter_length % 2 == 0
        half_window_length = (filter_length - 1) // 2

        # Determine the step size for down-sampling
        step_size = ds_factor

        # Mark the indices for calculating energy
        mark_indices = np.arange(
            1 + half_window_length, len(audio_data) - half_window_length, step_size
        )

        # Convert indices to time in seconds
        time_seconds = np.divide(np.subtract(mark_indices, 1), sample_rate)

        # Number of frames to process
        num_frames = len(mark_indices)

        # Preallocate array for energy values
        energy_values = np.zeros(num_frames)

        # Calculate the energy for each frame
        for i in range(num_frames):
            n = mark_indices[i]

            # Extract a window of data centered around the current mark
            signal_window = audio_data[n - half_window_length : n + half_window_length + 1]

            # Remove the DC component (mean value) from the signal
            signal_dc = signal_window - np.mean(signal_window)

            # Compute the root mean square (RMS) energy of the de-meaned signal
            energy_values[i] = np.sqrt(np.sum(signal_dc**2) / filter_length)

        # Combine time and energy into a 2D array
        energy_bp = np.column_stack((time_seconds, energy_values))

        return energy_bp


    def _getspec(
        self,
        audio_data: np.ndarray, 
        sample_rate: int, 
        hopsize: int, 
        n_fft: int,
        window_size: int, 
        window_type: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate STFT.

        Parameters
        ----------
        audio_data: numpy.ndarray
            Numpy array containing data samples
        sample_rate: int
            Sampling rate in Hz
        hopsize: int
            Hop size in samples
        n_fft: int
            Number of FFT points
        window: str
            Window function to apply to the signal

        Returns
        -------
        spec: numpy.ndarray
            A 2D array containing the spectrogram of the audio data

        """
        # Calculate the number of frames needed
        num_f = int(np.ceil(len(audio_data) / hopsize))
        # print("num_f: " + str(num_f))

        # Calculate padding 
        pad = - (len(audio_data) - num_f * hopsize)
        # print("pad: " + str(pad))

        # Pad the data with zeros if necessary
        if pad > 0:
            data = np.concatenate([audio_data, np.zeros(int(pad))])
        # print("data[200:210]: " + str(data[200:211]))

        w = None
        if window_type == "hamming":
            w = np.hamming(window_size)
        else:
            raise ValueError("Unsupported window type")
        # print("w.shape" + str(w.shape))
        # print("w[300:310]: " + str(w[300:311]))

        # Normalize the window function
        norm_window = np.sum(w ** 2)
        # print("norm_window: " + str(norm_window))

        # Compute the spectrogram using librosa's stft function instead of specgram2
        spec = librosa.stft(audio_data.flatten(), n_fft=n_fft, hop_length=hopsize, window=np.hamming(n_fft)/np.sqrt(n_fft * norm_window))

        # Get the magnitude of the FFT
        fftout = np.abs(spec)
        # print(fftout.shape)
        # print("fftout[300:304, 95:97]: " + str(fftout[300:304, 95:97]))

        # Get the phase of the FFT
        phase = np.angle(spec)
        # print("phase[300:304, 95:97]: " + str(phase[300:304, 95:97]))

        # Compensate for the zero-padding in the last frame if needed
        if pad > 0:
            fftout[:, -1] *= np.sqrt(window_size / (window_size - pad))
        # print("np.sqrt(window_size / (window_size - pad)): " + str(np.sqrt(window_size / (window_size - pad))))

        return (fftout, phase)

    def _specgram2():
        # todo port from matlab
        pass

    def _h_spectre(self):
        if self.mf0_hz is None:
            self.aff()
        L_sec = self.nbt0 / self.mf0_hz
        sr_hz = self.sample_rate
        data_v = self.audio_data
        overlap_factor = self.overlap_factor
        windowTYPE = self.window

        L_n = round(L_sec * sr_hz) + 1
        L_n = L_n + (L_n % 2 == 0)  # Ensuring L_n is odd
        LD_n = (L_n - 1) // 2
        N = 2 * 2 ** int(np.ceil(np.log2(L_n)))  # Next power of 2
        STEP_n = round(L_n / overlap_factor)

        window_v = hamming(L_n)
        normalisation = sum(window_v)

        mark_n_v = np.arange(1 + LD_n, len(data_v) - LD_n, STEP_n)
        nb_analyses = len(mark_n_v)
        
        X_m = np.zeros((N // 2 + 1, len(mark_n_v)))  # Initialize X_m with appropriate shape

        for frame in range(len(mark_n_v)):
            n = mark_n_v[frame]
            t = (n - 1) / sr_hz
            
            # Extract signal segment and apply mean centering and window
            signal_v = data_v[n - LD_n:n + LD_n + 1]  # Ensure indexing includes n + LD_n
            signal_v -= np.mean(signal_v)  # Mean centering
            signal_v *= window_v  # Apply window function
            
            # Perform FFT
            X_fft_iv = np.fft.fft(signal_v, N) / normalisation
            ampl_fft_v = np.abs(X_fft_iv[:N // 2])  # Get amplitude spectrum
            
            # Store results in X_m
            X_m[:N // 2 + 1, frame] = [t] + ampl_fft_v.tolist()[:N // 2]  # Combine time and amplitude

        self.X_m = X_m
