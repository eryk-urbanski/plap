import numpy as np


def amplitude_envelope(audio_signal, sample_rate, block_size, overlap):
    frame_length_samples = int(block_size * sample_rate)
    step = round((100 - overlap) / 100 * frame_length_samples)

    # Initialize the envelope
    envelope = np.zeros_like(audio_signal)
    current_max = 0

    for i in range(len(audio_signal)):
      
        if audio_signal[i] > current_max:
            current_max = audio_signal[i]
      
        elif i % step == 0:
            current_max *= (
                0.99 
            )

        envelope[i] = current_max

    return envelope


def rms(frames: np.ndarray) -> np.ndarray:
    
    rms_values = np.sqrt(np.mean(np.square(frames), axis=1))
    return rms_values


def zero_crossing_rate(frames):
    zcr_values = []
    for frame in frames:
        zcr = np.sum(np.abs(np.diff(np.sign(frame))) / 2)
        zcr_values.append(zcr)
    return np.array(zcr_values)

def temporal_centroid(audio_signal, sample_rate):
  
    weighted_sum = np.sum(np.arange(len(audio_signal)) * np.abs(audio_signal))
    amplitude_sum = np.sum(np.abs(audio_signal)) 
    tc_samples = weighted_sum / amplitude_sum if amplitude_sum != 0 else 0
    tc_seconds = tc_samples / sample_rate
    
    return tc_seconds
