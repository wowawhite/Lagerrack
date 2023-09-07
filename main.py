import numpy as np
import os.path
from scipy.io.wavfile import read
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import pywt
import scipy.io.wavfile
import pyqtgraph as pg
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QApplication
#import cupy as cp
#import cupyx.scipy.fft as cufft
import scipy.fft

work_dir= "C:\\Users\\User\\Desktop\\Testbetrieb\\snippets"
base_filename='nok1_test'
file_type=".wav"
target_file = base_filename + file_type
wavefile = os.path.join(work_dir, target_file)

# read the wavefile
# -> Data array is in signal as np array
sampling_frequency, signal = read(wavefile)

print(f"sampling_frequency = {sampling_frequency}")
print(f"shape[0] = {signal.shape[0]}")
length = signal.shape[0] / sampling_frequency
print(f"length = {length}")
scales = (1, len(signal))
print(f"scales = {scales}")
time = np.linspace(0., length, signal.shape[0]) # start, end, spacing
print(f"time = {time}")
sample_stepping = 1.0/sampling_frequency
N = 100000
#yNorm = signal / np.linalg.norm(signal)
yNorm = signal / np.max(signal)


# Plotting source signal
plt.subplot(211)
plt.plot(time[0:N], yNorm[0:N], label="source signal")
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid()

# Playing around

# FFT tryout
yf = fft(yNorm)[:N//2]
xf = fftfreq(N, sample_stepping)[:N//2]
yf = yf / np.max(yf)
plt.subplot(212)
plt.plot(xf, np.abs(yf), label="fft")
plt.legend()
plt.xlabel("Freq [Hz]")
plt.ylabel("Amplitude")
plt.grid()




#plot it!
plt.show()



# https://pywavelets.readthedocs.io/en/latest/ref/cwt.html
#coefficient, frequency = pywt.cwt(signal, scales, 'gaus1')
# DWT Decomposition
#cA, cD = pywt.dwt(signal[0:100], 'db1')
#Set thesholds

# DWT Reconstruction


#x, sr = librosa.load(librosa.example("libri3"))
#x = wavefile
#x = x[sr:2 * sr]
#x = x / np.max(np.abs(x))

exit()





