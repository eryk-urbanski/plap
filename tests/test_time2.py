import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.append("..")
from plap.core.audio_info import AudioInfo
from plap.core.preprocessing import Preprocessing
from plap.parameterization.time import rms

x = AudioInfo("data/001_guit_solo.wav")
frames = Preprocessing.framing(audio_info=x, block_size=512, overlap=0)
rms_values = rms(frames)

plt.figure(figsize=(15, 5))
plt.plot(np.repeat(rms_values, int(512 * (100 - 0) / 100)), label="RMS", color='red') 
plt.legend()
plt.title("Oryginalny sygnał i jego RMS")
plt.xlabel("Indeks próbki")
plt.ylabel("Amplituda")
plt.grid(True)
plt.show()