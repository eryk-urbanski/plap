import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.append("..")
from plap.core.audio_info import AudioInfo
from plap.core.preprocessing import Preprocessing
from plap.parameterization.time import amplitude_envelope

x = AudioInfo("data/001_guit_solo.wav")
Preprocessing.framing(audio_info=x, block_size=512, overlap=0)
env = amplitude_envelope(x.signal, x.sample_rate, 0.003, 80)

plt.figure(figsize=(15, 5))
plt.plot(x.signal, label="Original Signal", alpha=0.5)
plt.plot(env, label="Max-Hold Amplitude Envelope", color='red')
plt.legend()
plt.title("Original Signal and Max-Hold Amplitude Envelope")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()