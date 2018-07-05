import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy.io import wavfile
import sounddevice as sd

# OPEN THE SOUND FILE
print("Opening the sound file... \n")
sample_rate, data = wavfile.read('noisy_sound.wav')  # load the data
print("Opened successfully..\n")
print("Sampling Frequency of the signal = ", sample_rate)
samples = len(data)
print("Total samples = ", samples)
sd.play(data, sample_rate)

print("Data: ", data, data.dtype)
print("Range of amplitude: ", min(data), max(data))
plt.plot(data)
plt.title("Time domain")
plt.ylabel("Amplitude")
plt.xlabel("Time (samples)")
# plt.savefig("original_time_domain.png", bbox_inches='tight')
plt.show()

print("Computing FFT.. ")
freq_data = fft(data)
freq_data = freq_data
plt.plot(freq_data[:len(freq_data) // 2])
plt.title("Frequency domain")
plt.ylabel("Amplitude")
plt.xlabel("Frequency")
# plt.savefig("original_freq_domain.png", bbox_inches='tight')
plt.show()

# NOW REMOVE NOISE ==============================
freq_data[20000:] = 0
plt.plot(freq_data[:len(freq_data) // 2])
plt.title("Filtered Frequency domain")
plt.ylabel("Amplitude")
plt.xlabel("Frequency")
plt.savefig("filtered_freq_domain.png", bbox_inches='tight')
plt.show()



print("Computing Inverse FT.. ")
new_data = ifft(freq_data)
new_data = new_data.astype('int16')
print("New data:", new_data)
sd.play(new_data, sample_rate)
plt.plot(new_data)
plt.title("IFT Time domain")
plt.ylabel("Amplitude")
plt.xlabel("samples")
plt.savefig("ifft_time_domain.png", bbox_inches='tight')
plt.show()
print(len(new_data))
wavfile.write("denoise.wav", sample_rate, new_data)
