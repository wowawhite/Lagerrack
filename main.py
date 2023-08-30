import numpy as np
import os.path
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import pywt
import scipy.io.wavfile
import pyqtgraph as pg
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QApplication

work_dir= "C:\\Users\\User\\Desktop\\Testbetrieb\\snippets"
base_filename='nok1_test'
file_type=".wav"
target_file = base_filename + file_type
wavefile = os.path.join(work_dir, target_file)

# read the wavefile
sampling_frequency, signal = read(wavefile)
print(f"signal = {signal}")
print(f"number of channels = {signal.shape[0]}")
length = signal.shape[0] / sampling_frequency
print(f"length = {length}")

scales = (1, len(signal))
print(f"scales = {scales}")
time = np.linspace(0., length, signal.shape[0])
print(f"time = {time}")


plt.plot(time[0:100], signal[0:100], label="mono channel")
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.show()

# https://pywavelets.readthedocs.io/en/latest/ref/cwt.html
#coefficient, frequency = pywt.cwt(signal, scales, 'gaus1')
# DWT Decomposition
cA, cD = pywt.dwt(signal[0:100], 'db1')
#Set thesholds

# DWT Reconstruction

print(f"coefficient, frequency = {coefficient},{frequency}")
#x, sr = librosa.load(librosa.example("libri3"))
#x = wavefile
#x = x[sr:2 * sr]
#x = x / np.max(np.abs(x))

exit()





