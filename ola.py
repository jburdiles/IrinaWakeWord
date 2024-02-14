######## IMPORTS ##########
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import librosa.display
import matplotlib.pyplot as plt

filename = "audio_data\\36.wav"

data, sample_rate = librosa.load(filename)

plt.title("Wave Form Ola")
librosa.display.waveshow(data, sr=sample_rate)
plt.show()