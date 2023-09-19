#!/usr/bin/env python
import numpy as np
import os.path
import os
# import pywt
#from numba import jit

import timeit
from statsmodels.tsa.stattools import acf, pacf

from scipy.io.wavfile import read

from scipy.fft import fft, fftfreq
from scipy import signal
import matplotlib.pyplot as plt

rng = np.random.default_rng()

import pyqtgraph as pg
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QApplication



# import tensorflow as tf

# program control global variables
USE_CUDA = False
USE_PLOTGRAPH = True
USE_DEBUGPRINT = True
OS_TYPE = os.name

if (USE_DEBUGPRINT):
    print("Debug printing enabled")
    pass

if USE_CUDA:
    print("Enable CUDA Support")
    import pycuda.autoinit
    import pycuda.driver as drv
    import cupy as cp  # install pip cupy-cuda12x
    import cupyx as cpx

    # import cusignal as cus  # needs WSL2 (yes, not WSL1)
    CUPY_GPU_MEMORY_LIMIT = "4294967296"  # "1073741824"
    # do not use  scipy.fft!
    feedback_cuda = cp.cuda.runtime.driverGetVersion()
    print(f"CUDA driver version is {feedback_cuda}")


def normalizing(nsignal):
    y_norm = nsignal / np.max(nsignal)
    return y_norm


def timecropping(signal_length, sampling_freq, full_signal, time_limit):
    if signal_length < timelimit:
        time_limit = signal_length
    cropped_N = int(sampling_freq * timelimit)
    print(f"cropping {time_limit} seconds and {cropped_N} samples")
    cropped_signal = full_signal[0: cropped_N]
    cropped_length = cropped_signal.shape[0] / sampling_freq
    return cropped_length, cropped_N, cropped_signal


def read_sourcefile(filename, croptime=None):
    match OS_TYPE:
        case "nt":
            work_dir = "C:\\Users\\User\\Desktop\\Testbetrieb\\snippets"
        case "posix":
            myuser = os.environ.get('USER')
            work_dir = "//home//"+myuser+"//Schreibtisch//Testbetrieb//snippets"
        case _:
            work_dir = "C:\\Users\\User\\Desktop\\Testbetrieb\\snippets"
    base_filename = filename
    file_type = ".wav"
    target_file = base_filename + file_type
    wavefile = os.path.join(work_dir, target_file)
    sampling_frequency, full_signal = read(wavefile)
    sig_length = full_signal.shape[0] / sampling_frequency
    if (USE_DEBUGPRINT):
        print(f"Opening file:{wavefile}")
        print(f"sampling_frequency = {sampling_frequency}")
        print(f"shape[0] = {full_signal.shape[0]}")
        print(f"full signal length  = {sig_length} [s]")
    cropped_length, cropped_N, cropped_signal = \
        timecropping(sig_length, sampling_frequency, full_signal, croptime)
    cropped_normalized_signal = normalizing(cropped_signal)
    return sampling_frequency, cropped_normalized_signal


def calculate_passfilter(fft_sampling_frequency, fft_source_signal, low, high):
    return


def calculate_spectraldensity(sd_sampling_frequency, sd_source_signal):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html
    fs = 10e3
    N = 1e5
    amp = 2 * np.sqrt(2)
    freq = 1234.0
    noise_power = 0.001 * fs / 2
    time_t = np.arange(N) / fs
    x = amp * np.sin(2 * np.pi * freq * time_t)
    x += rng.normal(scale=np.sqrt(noise_power), size=time_t.shape)
    f, Pxx_den = signal.periodogram(x, fs)
    print(f"f, Pxx_den  = {f} - {Pxx_den} [s]")
    return f, Pxx_den


def calculate_fft(fft_sampling_frequency, fft_source_signal):
    N = np.size(fft_source_signal)
    sample_stepping = 1.0 / fft_sampling_frequency
    xf = fftfreq(N, sample_stepping)[:N // 2]

    if USE_CUDA:
        cache = cp.fft.config.get_plan_cache()
        cache.clear()

        cp._default_memory_pool.free_all_blocks()
        print("Used GPU cache before calculation:", cp.get_default_memory_pool().used_bytes() / 1024, "kB")
        cp.cuda.runtime.memGetInfo()
        cyNorm = cp.array(fft_source_signal)
        cyf = cp.fft.fft(cyNorm)[:N // 2]
        yf = cp.asnumpy(cyf)
        # flush GPU memory to avoid cuda memory leak
        cache = cp.fft.config.get_plan_cache()
        cache.clear()
        print("Used GPU cache after calculation:", cp.get_default_memory_pool().used_bytes() / 1024, "kB")

    else:
        yfn = fft(fft_source_signal)
        yf = yfn[:N // 2]

    yf = yf / np.max(yf)

    return xf, yf


def calculate_autocorrelation(sampling_frequency, acf_signal):
    if USE_CUDA:
        pass  # TODO autocorrelation cuda
    else:

        # https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acf.html#statsmodels.tsa.stattools.acf
        yf = acf(acf_signal)
        print(f"size yf = {np.size(yf)}")
        print(f"shape yf = {yf.shape[0]}")
        xf = np.arange(0, yf.shape[0])
        print(f"shape xf = {xf.shape[0]}")
        print(yf)
        # alternative fft acf
    return xf, yf


def calculate_autocorrelation2(sampling_frequency, acf_signal):
    result = np.correlate(acf_signal, acf_signal, mode='full')

    yf = result[result.size / 2:]
    xf = np.arange(0, yf.shape[0])
    return xf, yf


def calculate_discretewavelet():
    # https://pywavelets.readthedocs.io/en/latest/ref/cwt.html
    # coefficient, frequency = pywt.cwt(signal, scales, 'gaus1')
    # DWT Decomposition
    # cA, cD = pywt.dwt(signal[0:100], 'db1')
    # Set thesholds

    # DWT Reconstruction

    # x, sr = librosa.load(librosa.example("libri3"))
    # x = wavefile
    # x = x[sr:2 * sr]
    # x = x / np.max(np.abs(x))

    return


if __name__ == "__main__":
    timelimit = 50.0  # Limit is 400s due GPU low memory size
    sampling_frequency1, signal1 = read_sourcefile("ok2", timelimit)
    sampling_frequency2, signal2 = read_sourcefile("nok2", timelimit)

    time = np.linspace(0., signal1.shape[0] / sampling_frequency1, signal1.shape[0])  # start, end, spacing
    time2 = np.linspace(0., signal2.shape[0] / sampling_frequency2, signal2.shape[0])  # start, end, spacing

    # fft_ok1 = calculate_fft(sampling_frequency, signal)
    # fft_nok1 = calculate_fft(sampling_frequency2, signal2)

    SD_ok1, ok = calculate_spectraldensity(sampling_frequency1, signal1)
    # SD_nok1 = calculate_spectraldensity(sampling_frequency2, signal2)

    # calculate_autocorrelation(signal_length, sampling_frequency, signal, timelimit)
    # calculate_autocorrelation(signal_length2, sampling_frequency2, signal2, timelimit)

    if USE_PLOTGRAPH:
        # TODO create an array with calculations and add them to subplot by iteration
        fig, (axs1) = plt.subplots(1)
        # fig, (axs3,axs4) = plt.subplots()
        # axs1.plot(time, signal, label="source signal1")
        # axs1.xlabel("Time [s]")
        # axs1.ylabel("Amplitude")

        # axs2.plot(time2, signal2, label="source signal2")
        # axs2.xlabel("Time [s]")
        # axs2.ylabel("Amplitude")

        # axs3.plot(fft_ok1[0], np.abs(fft_ok1[1]), label="fft_signal1")
        # axs3.xlabel("Freq [Hz]")
        # axs3.ylabel("Amplitude")

        # axs4.plot(fft_nok1[0], (fft_ok1[1]), label="fft_signal2")
        # axs4.xlabel("Freq [Hz]")
        # axs4.ylabel("Amplitude")
        # axs1.semilogx()
        # axs1.semilogy()
        # axs1.set_ylim([1e-7, 1e2])
        # axs1.plot(SD_ok1[0], (SD_ok1[1]), label="fft_signal2")
        # axs2.plot(SD_nok1[0], (SD_nok1[1]), label="fft_signal2")

        plt.semilogy(SD_ok1, ok)
        plt.ylim([1e-7, 1e2])
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        plt.show()
