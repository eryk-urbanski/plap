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
    #  - Harmonic Spectral Centroid Descriptor HSC TODO COMPARISON
    #  - Harmonic Spectral Deviation Descriptor HSD TODO COMPARISON
    #  - Harmonic Spectral Spread Descriptor HSS TODO COMPARISON
    #  - Harmonic Spectral Variation Descriptor HSV TODO COMPARISON
    # Spectral Basis
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
        self.aff_d = None
        self.mf0_hz = None
        self.nbt0 = 8
        self.overlap_factor = 2
        self.L_sec = None
        self.X_m = None # STFT
        self.pic_struct = None
        # Harmonic Descriptors
        self.harmo_called = False
        self.hsc_d = None
        self.hsd_d = None
        self.hss_d = None
        self.hsv_d = None

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

        if self.aff_d is not None:
            return self.aff_d

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
                temp = np.sqrt(np.clip(den1 * den, 0, None))
                phi[k - Kl] = num / temp

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

        self.aff_d = f0
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
        print(f"t[stop], t[start]: {time_v[stopattack_pos], time_v[startattack_pos]}")
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
    
    # Harmonic Spectral Descriptors
    # -> HSC, HSS
    def hsdescriptors(self):
        harmonic_spectral_centroid = 0
        harmonic_spectral_spread = 0
        if self.X_m is None:
            self._h_spectre()
        pic_struct = self._h_harmo()
        harmonic_spectral_centroid, harmonic_spectral_spread = self._h_harmoiParam(pic_struct=pic_struct)
        return harmonic_spectral_centroid, harmonic_spectral_spread
    
    # Harmonic Spectral Centroid HSC
    def hsc(self):
        harmonic_spectral_centroid = 0
        if self.X_m is None:
            self._h_spectre()
        if self.pic_struct is None:
            self._h_harmo()
        pic_struct = self.pic_struct
        if self.hsc_d is None:
            harmonic_spectral_centroid, _, _, _ = self._h_harmoiParam(pic_struct=pic_struct)
        else:
            harmonic_spectral_centroid = self.hsc_d
        return harmonic_spectral_centroid
    
    # Harmonic Spectral Deviation HSD
    def hsd(self):
        harmonic_spectral_deviation = 0
        if self.X_m is None:
            self._h_spectre()
        if self.pic_struct is None:
            self._h_harmo()
        pic_struct = self.pic_struct
        if self.hsd_d is None:
            _, harmonic_spectral_deviation, _, _ = self._h_harmoiParam(pic_struct=pic_struct)
        else:
            harmonic_spectral_deviation = self.hsd_d
        return harmonic_spectral_deviation
    
    # Harmonic Spectral Spread HSS
    def hss(self):
        harmonic_spectral_spread = 0
        if self.X_m is None:
            self._h_spectre()
        if self.pic_struct is None:
            self._h_harmo()
        pic_struct = self.pic_struct
        if self.hss_d is None:
            _, _, harmonic_spectral_spread, _ = self._h_harmoiParam(pic_struct=pic_struct)
        else:
            harmonic_spectral_spread = self.hss_d
        return harmonic_spectral_spread
    
    # Harmonic Spectral Variation HSV
    def hsv(self):
        harmonic_spectral_variation = 0
        if self.X_m is None:
            self._h_spectre()
        if self.pic_struct is None:
            self._h_harmo()
        pic_struct = self.pic_struct
        if self.hsv_d is None:
            _, _, _, harmonic_spectral_variation = self._h_harmoiParam(pic_struct=pic_struct)
        else:
            harmonic_spectral_variation = self.hsv_d
        return harmonic_spectral_variation

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

        # Replace frequencies less than 62.5Hz with nominal freq 31.25Hz
        num_less_low_freq = np.sum(fft_freq < low_freq)
        fft_freq = np.concatenate(([31.25], fft_freq[num_less_low_freq:]))

        # Log-scaled frequencies relative to 1kHz
        fft_freq_log = np.log2(fft_freq / 1000)

        # Calculate powers !!! something off, check again after implementing specgram2 in getspec instead of librosa
        powers = fftout**2
        powers[1:-1, :] = 2 * powers[1:-1, :]

        # Sum powers for the frequencies below loedge
        if num_less_low_freq > 1:
            summed_powers = np.sum(powers[:num_less_low_freq, :], axis=0, keepdims=True)
            powers = np.concatenate((summed_powers, powers[num_less_low_freq:, :]), axis=0)

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
        # if pad > 0:
        #     data = np.concatenate([audio_data, np.zeros(int(pad))])
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

    def _h_harmo(self):
        crible = 0.1
        f0_bp = self.f0_bp
        if self.X_m is None:
            self._h_spectre()
        X_m = self.X_m
        nb_frames = X_m.shape[1]
        nsams = X_m.shape[0] - 1
        N = nsams * 2

        pic_struct = []  # Initialize an empty list to store frame data

        for frame in range(nb_frames):
            sr_hz = self.sample_rate
            # Extract time and amplitude values for the current frame
            t = X_m[0, frame]
            ampl_fft_v = X_m[1:int(N / 2) + 1, frame]
            
            # Calculate f0 (fundamental frequency) at time t
            f0_hz = self._h_evalbp(f0_bp, t)
            
            # Calculate the harmonic count H
            H = round(0.5 * sr_hz / f0_hz)
            
            # Get harmonic frequencies and amplitudes
            freqh_hz_v, amplh_lin_v = self._h_harmopic(H, f0_hz, crible, ampl_fft_v, sr_hz, N)
            
            # Store results in pic_struct
            pic_struct.append({
                'freqh_v': freqh_hz_v,
                'amplh_lin_v': amplh_lin_v
            })
        
        self.pic_struct = pic_struct

    def _h_evalbp(self, bp, t):
        # if len(t) != 1:
        #     raise ValueError("_h_evalbp: length of t is not 1")
        pos = np.argmin(np.abs(bp[:, 0] - t))
        taille = bp.shape
        if (bp[pos, 0] == t) or (taille[0] == 1) or \
        ((bp[pos, 0] < t) and (pos == taille[0] - 1)) or \
        ((bp[pos, 0] > t) and (pos == 0)):
            
            value = bp[pos, 1]

        elif bp[pos, 0] < t:
            # Linear interpolation if bp[pos, 0] < t
            value = (bp[pos + 1, 1] - bp[pos, 1]) / (bp[pos + 1, 0] - bp[pos, 0]) * (t - bp[pos, 0]) + bp[pos, 1]

        elif bp[pos, 0] > t:
            # Linear interpolation if bp[pos, 0] > t
            value = (bp[pos, 1] - bp[pos - 1, 1]) / (bp[pos, 0] - bp[pos - 1, 0]) * (t - bp[pos - 1, 0]) + bp[pos - 1, 1]

        return value
    
    def _h_harmopic(self, H, f0_hz, c, am_fft_v, sr_hz, N):
        harmo_hz_v = np.arange(1, H + 1) * f0_hz  # Create harmonic frequencies
        freqh_hz_v = np.zeros(H)                  # Initialize frequency array
        amplh_lin_v = np.zeros(H)                 # Initialize amplitude array

        for h in range(H):
            # Define the frequency zone for the current harmonic
            zone_hz = np.arange(max(0, harmo_hz_v[h] - c * f0_hz),
                                min(harmo_hz_v[h] + c * f0_hz, sr_hz / 2 - sr_hz / N), sr_hz / N)
            
            # Convert the frequency zone to sample indices
            zone_k = np.round(zone_hz / sr_hz * N).astype(int)
            
            if len(zone_k) > 0:
                # Find the maximum amplitude within this zone
                max_value = np.max(am_fft_v[zone_k])
                max_pos = zone_k[np.argmax(am_fft_v[zone_k])]
                
                # Store the frequency and amplitude
                freqh_hz_v[h] = (max_pos - 1) / N * sr_hz
                amplh_lin_v[h] = max_value

        # Filter out zero entries in freqh_hz_v to get the actual number of harmonics found
        pos_v = np.where(freqh_hz_v > 0)[0]
        H = pos_v[-1] + 1 if len(pos_v) > 0 else 0
        freqh_hz_v = freqh_hz_v[:H]
        amplh_lin_v = amplh_lin_v[:H]

        return freqh_hz_v, amplh_lin_v

    def _h_harmoiParam(self, pic_struct):
        self.harmo_called = True
        nb_frames = len(pic_struct)
        inrg = np.zeros(nb_frames)
        iHarmonicSpectralCentroid = np.zeros(nb_frames)
        iHarmonicSpectralDeviation = np.zeros(nb_frames)
        iHarmonicSpectralSpread = np.zeros(nb_frames)
        iHarmonicSpectralVariation = np.zeros(nb_frames)

        # Instantaneous values
        for frame in range(nb_frames):
            freqh_v = pic_struct[frame]['freqh_v']
            amplh_lin_v = pic_struct[frame]['amplh_lin_v']
            H = len(freqh_v)
            
            # inrg
            inrg[frame] = np.sqrt(np.sum(amplh_lin_v[:H] ** 2))
            
            # ihsc, ihss
            iHarmonicSpectralCentroid[frame], iHarmonicSpectralSpread[frame] = self._ihsc_ihss(freqh_v, amplh_lin_v, H)

            # ihsd
            SE_lin_v = self._spectral_env(amplh_lin_v=amplh_lin_v)
            iHarmonicSpectralDeviation[frame] = self._ihsd(np.log(np.clip(amplh_lin_v, np.finfo(float).eps, None)), np.log(np.clip(SE_lin_v, np.finfo(float).eps, None)), H)

            # ihsv
            if frame > 0:
                # Determine the minimum length between current and previous frame harmonics
                minH = min(H_old, H)
                iHarmonicSpectralVariation[frame] = self._ihsv(
                    amplh_lin_v_old[:minH], amplh_lin_v[:minH], minH
                )

            amplh_lin_v_old = amplh_lin_v
            H_old = H


        # Find positions where `inrg` is greater than 10% of its maximum value
        pos_v = np.where(inrg > np.max(inrg) * 0.1)[0]  # `np.where` returns a tuple, so we use [0] to get the indices
        # Calculate the mean of `iHarmonicSpectralCentroid` at the positions found
        self.hsc_d = np.mean(iHarmonicSpectralCentroid[pos_v]) if len(pos_v) > 0 else 0
        self.hsd_d = np.mean(iHarmonicSpectralDeviation[pos_v]) if len(pos_v) > 0 else 0
        self.hss_d = np.mean(iHarmonicSpectralSpread[pos_v]) if len(pos_v) > 0 else 0
        self.hsv_d = np.mean(iHarmonicSpectralVariation[pos_v]) if len(pos_v) > 0 else 0
        return self.hsc_d, self.hsd_d, self.hss_d, self.hsv_d

    def _ihsc_ihss(self, freqh_v, amplh_v, H):
        if len(freqh_v) < H or len(amplh_v) < H:
            raise ValueError("_hsc")
        # Ensure freqh_v and amplh_v are column vectors
        freqh_v = np.reshape(freqh_v, (-1, 1))
        amplh_v = np.reshape(amplh_v, (-1, 1))

        # ihsc computing
        num = np.sum(freqh_v[:H] * amplh_v[:H])
        denum = np.sum(amplh_v[:H])
        denum = denum + np.finfo(float).eps if denum == 0 else denum
        HarmonicSpectralCentroid = num / denum

        # ihss computing
        num = np.sum((amplh_v[:H] * (freqh_v[:H] - HarmonicSpectralCentroid))**2)
        denum = np.sum(amplh_v[:H]**2)
        denum = denum + np.finfo(float).eps if denum == 0 else denum
        hsc_temp = HarmonicSpectralCentroid + np.finfo(float).eps if HarmonicSpectralCentroid == 0 else HarmonicSpectralCentroid
        HarmonicSpectralSpread = (1 / hsc_temp) * np.sqrt(num / denum)

        return HarmonicSpectralCentroid, HarmonicSpectralSpread

    def _ihsv(self, x1_v, x2_v, H):
        # Error check equivalent: ensuring both vectors are at least of length H
        if len(x1_v) < H or len(x2_v) < H:
            raise ValueError("_ihsv: Length of vectors is less than H")
        # Calculate the cross product of the vectors
        crossprod = np.sum(x1_v[:H] * x2_v[:H])

        # Calculate the auto-products of each vector
        autoprod_x1 = np.sum(x1_v[:H] ** 2)
        autoprod_x2 = np.sum(x2_v[:H] ** 2)

        # Compute the Harmonic Spectral Variation
        temp = autoprod_x2 * autoprod_x1 
        temp = 0 if temp < 0 else temp
        HarmonicSpectralVariation = 1 - crossprod / (np.sqrt(temp))

        return HarmonicSpectralVariation

    def _spectral_env(self, amplh_lin_v):
        # Ensure the input is a column vector
        amplh_lin_v = np.asarray(amplh_lin_v).flatten()
        H = len(amplh_lin_v)
        SE_lin_v = np.zeros(H)
        # === Spectral envelope estimation
        # First element
        SE_lin_v[0] = (amplh_lin_v[0] + amplh_lin_v[1]) / 2

        # Middle elements
        for kk in range(1, H - 1):
            SE_lin_v[kk] = (amplh_lin_v[kk - 1] + amplh_lin_v[kk] + amplh_lin_v[kk + 1]) / 3

        # Last element
        SE_lin_v[-1] = (amplh_lin_v[-2] + amplh_lin_v[-1]) / 2

        return SE_lin_v
    
    def _ihsd(self, amplh_v, SE_v, H):
        # Check if the length of input vectors is sufficient
        if len(amplh_v) < H or len(SE_v) < H:
            raise ValueError("_ihsd: Length of input vectors is less than H")

        # Ensure the input vectors are column vectors
        amplh_v = np.asarray(amplh_v).flatten()
        SE_v = np.asarray(SE_v).flatten()

        # Harmonic Spectral Deviation computation
        harmonic_spectral_deviation = np.sum(np.abs(amplh_v[:H] - SE_v[:H]))

        nb_harmo = H
        harmonic_spectral_deviation /= nb_harmo

        return harmonic_spectral_deviation