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



#import tensorflow as tf

# program control global variables
USE_CUDA = False
USE_PLOTGRAPH = True
USE_DEBUGPRINT = True




if(USE_DEBUGPRINT):
    print("Debug printing enabled")
    pass

if USE_CUDA:
    print("Enable CUDA Support")
    import cupy as cp  # install pip cupy-cuda12x
    import cupyx as cpx
    import cusignal as cus  # needs WSL2 (yes, not WSL1)

    # do not use  scipy.fft!
    cp.cuda.runtime.driverGetVersion()
    pass

def read_sourcefile():
    work_dir = "C:\\Users\\User\\Desktop\\Testbetrieb\\snippets"
    base_filename = 'ok1'
    file_type = ".wav"
    target_file = base_filename + file_type
    wavefile = os.path.join(work_dir, target_file)
    sampling_frequency, signal = read(wavefile)
    signal_length = signal.shape[0] / sampling_frequency

    if (USE_DEBUGPRINT):
        print(f"sampling_frequency = {sampling_frequency}")
        print(f"shape[0] = {signal.shape[0]}")
        print(f"length  = {signal_length} [s]")

    return signal_length, sampling_frequency, signal


def normalizing(signal):
    yNorm = signal / np.max(signal)
    return yNorm


def calculate_fft(signal_length, sampling_frequency, signal, timelimit):
    if signal_length < timelimit:
        timelimit = signal_length
    N = int(sampling_frequency * timelimit)
    print(f"cropping {timelimit} seconds and {N} samples")
    #signal_length = signal.shape[0] / sampling_frequency
    time = np.linspace(0., signal_length, signal.shape[0])  # start, end, spacing
    sample_stepping = 1.0/sampling_frequency
    #TODO: Crop time of signal to limit RAM usage



    normalized_signal = normalizing(signal)
    xf = fftfreq(N, sample_stepping)[:N // 2]

    if USE_CUDA:
        cache = cp.fft.config.get_plan_cache()
        cache.clear()

        cp._default_memory_pool.free_all_blocks()
        print("Used GPU cache before calculation:", cp.get_default_memory_pool().used_bytes() / 1024, "kB")
        cp.cuda.runtime.memGetInfo()
        cyNorm = cp.array(normalized_signal[0:N])
        cyf = cp.fft.fft(cyNorm)[:N//2]
        yf = cp.asnumpy(cyf)
        # flush GPU memory to avoid cuda memory leak
        cache = cp.fft.config.get_plan_cache()
        cache.clear()  # TODO: cache flush seems to have no effect
        print("after clearing cache:", cp.get_default_memory_pool().used_bytes() / 1024, "kB")

    else:
        yfn = fft(normalized_signal[0:N])
        yf = yfn[:N//2]

    yf = yf / np.max(yf)
    if USE_PLOTGRAPH:
        plt.subplot(211)
        plt.plot(time[0:N], normalized_signal[0:N], label="source signal")
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.subplot(212)
        plt.plot(xf, np.abs(yf), label="fft")
        plt.legend()
        plt.xlabel("Freq [Hz]")
        plt.ylabel("Amplitude")
        plt.grid()
        #plot it!
        plt.show()

        return






def calculate_dwt():

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
    pass


if __name__ == "__main__":
    signal_length, sampling_frequency, signal = read_sourcefile()
    timelimit = 5.0
    calculate_fft(signal_length, sampling_frequency, signal, timelimit)
    pass




