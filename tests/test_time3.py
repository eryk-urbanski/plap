import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.append("..")
from plap.core.audio_info import AudioInfo
from plap.core.preprocessing import Preprocessing
from plap.parameterization.time import zero_crossing_rate

x = AudioInfo("data/001_guit_solo.wav")
frames = Preprocessing.framing(audio_info=x, block_size=512, overlap=0)
zcr_values = zero_crossing_rate(frames)
print("Zero Crossing Rate dla każdej ramki:")
for i, zcr in enumerate(zcr_values):
    print(f"Ramka {i+1}: {zcr}")