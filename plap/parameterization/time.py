import numpy as np


def amplitude_envelope(audio_signal, sample_rate, block_size, overlap):
    frame_length_samples = int(block_size * sample_rate)
    step = round((100 - overlap) / 100 * frame_length_samples)

    # Initialize the envelope
    envelope = np.zeros_like(audio_signal)
    current_max = 0

    for i in range(len(audio_signal)):
        # Update the current max if the current sample is greater
        if audio_signal[i] > current_max:
            current_max = audio_signal[i]
        # Decay the current max based on the step
        elif i % step == 0:
            current_max *= (
                0.99  # Slight decay to allow the envelope to follow the signal down
            )

        envelope[i] = current_max

    return envelope


def rms(frames):
    rms_values = []
    for frame in frames:
        rms_value = np.sqrt(np.mean(np.square(frame)))
        rms_values.append(rms_value)
    return np.array(rms_values)


def zero_crossing_rate(frames):
    zcr_values = []
    for frame in frames:
        zcr = np.sum(np.abs(np.diff(np.sign(frame))) / 2)
        zcr_values.append(zcr)
    return np.array(zcr_values)
