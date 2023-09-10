import numpy as np
import os.path
#import pywt

import timeit

from scipy.io.wavfile import read
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

import pyqtgraph as pg
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QApplication

import pycuda.autoinit
import pycuda.driver as drv

import cupy as cp # install pip cupy-cuda12x
#import cupyx.scipy.fft as cufft
# do not use  scipy.fft!

import tensorflow as tf

work_dir= "C:\\Users\\User\\Desktop\\Testbetrieb\\snippets"
base_filename='ok1'
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
N = 1000
#yNorm = signal / np.linalg.norm(signal)
yNorm = signal / np.max(signal)

cyNorm = cp.array(yNorm)

# Plotting source signal
plt.subplot(211)
plt.plot(time[0:N], yNorm[0:N], label="source signal")
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid()

# Playing around

# FFT tryout

cyf = cp.fft.fft(cyNorm)[:N//2] # TODO: WTF?
xf = fftfreq(N, sample_stepping)[:N//2]
yf = cp.asnumpy(cyf)
yf = yf / np.max(yf)

plt.subplot(212)
plt.plot(xf, np.abs(yf), label="fft")
plt.legend()
plt.xlabel("Freq [Hz]")
plt.ylabel("Amplitude")
plt.grid()




#plot it!
plt.show()

# flush GPU memory to avoid cuda memory leak
cache = cp.fft.config.get_plan_cache()
cache.clear()
print("after clearing cache:", cp.get_default_memory_pool().used_bytes()/1024, "kB")


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





