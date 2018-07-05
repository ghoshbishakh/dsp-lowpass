import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy.io import wavfile
import sounddevice as sd

# OPEN THE SOUND FILE
print("Opening the sound file... \n")
sample_rate, data = wavfile.read('emma.wav')  # load the data
data = data.T[0]  # select a single channel
print("Opened successfully..\n")
print("Sampling Frequency of the signal = ", sample_rate)
print("Total samples = ", len(data))
sd.play(data, sample_rate)

print("Data: ", data, data.dtype)
print("Range of amplitude: ", min(data), max(data))
plt.plot(data)
plt.title("Time domain")
plt.ylabel("Amplitude")
plt.xlabel("Time (samples)")
plt.savefig("original_time_domain.png", bbox_inches='tight')
plt.show()

print("Computing FFT.. ")
freq_data = fft(data)
freq_data = freq_data
plt.plot(freq_data[:len(freq_data) // 2])
plt.title("Frequency domain")
plt.ylabel("Amplitude")
plt.xlabel("Frequency")
plt.savefig("original_freq_domain.png", bbox_inches='tight')
plt.show()

# NOW ADD NOISE ==============================

m, s = 0, 0.1  # mean and standard deviation
noise = np.random.normal(0, 800, 384752)
plt.plot(noise)
plt.title("Noise")
plt.show()

noisy_data = data + noise
noisy_data = noisy_data.astype('int16')

wavfile.write("noisy_sound.wav", sample_rate, noisy_data)

sd.play(noisy_data, sample_rate)

plt.plot(noisy_data)
plt.title("Noise added - Time domain")
plt.ylabel("Amplitude")
plt.xlabel("Time (samples)")
plt.savefig("noisy_time_domain.png", bbox_inches='tight')
plt.show()


# CALCULATE FFT OF NOISY SIGNAL
print("Computing FFT.. for noisy signal")
noisy_freq_data = fft(noisy_data)
plt.plot(noisy_freq_data[:len(noisy_freq_data) // 2])
plt.title("Noise added Frequency domain")
plt.ylabel("Amplitude")
plt.xlabel("Frequency")
plt.savefig("noisy_freq_domain.png", bbox_inches='tight')
plt.show()


# print("Computing Inverse FT.. ")
# new_data = ifft(noisy_freq_data)
# new_data = new_data.astype('int16')
# print("New data:", new_data)
# sd.play(new_data, sample_rate)
# plt.plot(new_data)
# plt.title("IFT Time domain")
# plt.ylabel("Amplitude")
# plt.xlabel("samples")
# plt.savefig("ifft_time_domain.png", bbox_inches='tight')
# plt.show()
# print(len(new_data))
# wavfile.write("denoise.wav", sample_rate, new_data)
