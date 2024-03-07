import librosa
from python_speech_features import mfcc
import numpy as np
import matplotlib.pyplot as plt
from spafe.features.mfcc import mfcc as spafe_mfcc


# Function to load audio file
def load_audio(file_path):
    signal, sample_rate = librosa.load(file_path)
    return signal, sample_rate

# Function to calculate MFCC using Librosa
def calculate_mfcc_librosa(signal, sample_rate):
    mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13, n_mels=20, hop_length=int(sample_rate*0.01), n_fft=int(sample_rate*0.025))
    return mfccs

# Function to calculate MFCC using python_speech_features
def calculate_mfcc_python_speech_features(signal, sample_rate):
    mfccs = mfcc(signal, samplerate=sample_rate, numcep=13, nfilt=20, winlen=0.025, winstep=0.01, nfft=551)
    return mfccs

# Function to calculate MFCC using spafe
def calculate_mfcc_spafe(signal, sample_rate):
    mfccs = spafe_mfcc(signal, fs=sample_rate, num_ceps=13)
    return mfccs

# Load the audio file
audio_file = "/Users/natalianelke/Desktop/pythonProject2/plap/tests/data/001_guit_solo.wav"
signal, sample_rate = load_audio(audio_file)

# Calculate MFCC using Librosa
mfccs_librosa = calculate_mfcc_librosa(signal, sample_rate)

# Calculate MFCC using python_speech_features
mfccs_python_speech_features = calculate_mfcc_python_speech_features(signal, sample_rate)

# Calculate MFCC using spafe
mfccs_spafe = calculate_mfcc_spafe(signal, sample_rate)

print("MFCC obliczone za pomocą Librosa:")
print(mfccs_librosa)
print("\nMFCC obliczone za pomocą python_speech_features:")
print(mfccs_python_speech_features)
print("\nMFCC obliczone za pomocą spafe:")
print(mfccs_spafe)

plt.figure(figsize=(15, 10))

# Plot MFCC using Librosa
plt.subplot(221)
librosa.display.specshow(mfccs_librosa, x_axis='time', sr=sample_rate)
plt.colorbar()
plt.title('MFCC Librosa')
plt.xlabel('Time')
plt.ylabel('MFCC Index')
plt.gca().set_xlim([0, len(signal)/sample_rate])

# Plot MFCC using python_speech_features
plt.subplot(222)
plt.imshow(np.swapaxes(mfccs_python_speech_features, 0, 1), cmap='viridis', origin='lower', aspect='auto', extent=[0, len(signal)/sample_rate, 0, 13])
plt.colorbar()
plt.title('MFCC python_speech_features')
plt.xlabel('Time')
plt.ylabel('MFCC Index')

# Plot MFCC using spafe
plt.subplot(223)
plt.imshow(mfccs_spafe.T, cmap='viridis', origin='lower', aspect='auto', extent=[0, len(signal)/sample_rate, 0, len(mfccs_spafe[0])])
plt.colorbar()
plt.title('MFCC spafe')
plt.xlabel('Time')
plt.ylabel('MFCC Index')

plt.tight_layout()
plt.show()