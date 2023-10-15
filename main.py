#!/usr/bin/env python
import pathlib

import numpy as np
import os.path
import os
from pathlib import Path
# import pywt


import timeit
from statsmodels.tsa.stattools import acf, pacf

from scipy.io.wavfile import read

from scipy.fft import fft, fftfreq
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib import gridspec

rng = np.random.default_rng()

import pyqtgraph as pg
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QApplication



# import tensorflow as tf

# program control global variables
USE_CUDA = True
USE_PLOTGRAPH = True
USE_DEBUGPRINT = False
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

    # import cusignal as cus  # needs WSL2 (yes, not WSL1) to run on windows
    CUPY_GPU_MEMORY_LIMIT = "4294967296"  # "1073741824"
    # do not use  scipy.fft!
    feedback_cuda = cp.cuda.runtime.driverGetVersion()
    print(f"CUDA driver version is {feedback_cuda}")


def normalizing(nsignal):
    y_norm = nsignal / np.max(nsignal)
    return y_norm


def timecropping(signal_length, sampling_freq, full_signal, start_time, time_len):
    if signal_length-time_len < time_len:
        start_time = signal_length - time_len
    #signal_N = int(sampling_freq * signal_length)
    first_N = int(sampling_freq * start_time)
    last_N = int(sampling_freq * time_len + first_N)
    count_N = last_N-first_N
    print(f"cropping from {start_time} to {start_time+time_len} seconds and {count_N} samples")
    cropped_signal = full_signal[first_N: last_N]
    cropped_length = cropped_signal.shape[0] / sampling_freq  # back to time
    return cropped_length, count_N, cropped_signal


def read_sourcefile(filename, starttime=None, croptime=None):
    parent = Path(__file__).resolve().parent
    if USE_DEBUGPRINT:
        print(parent)

    match OS_TYPE:
        case "nt":
            work_dir = ("\\MA_Testdaten\\snippets\\")
        case "posix":
            #myuser = os.environ.get('USER')
            work_dir = ("/MA_Testdaten/snippets/")
        case _:
            work_dir = "C:\\Users\\User\\Desktop\\Testbetrieb\\snippets"
    file_type = ".wav"
    wavefile = ( str(parent) + work_dir + filename + file_type)
    print(wavefile)
    sampling_frequency, full_signal = read(wavefile)
    sig_length = full_signal.shape[0] / sampling_frequency
    if (USE_DEBUGPRINT):
        print(f"Opening file:{wavefile}")
        print(f"sampling_frequency = {sampling_frequency}")
        print(f"shape[0] = {full_signal.shape[0]}")
        print(f"full signal length  = {sig_length} [s]")
    cropped_length, cropped_N, cropped_signal = \
        timecropping(sig_length, sampling_frequency, full_signal, starttime, croptime)
    cropped_normalized_signal = normalizing(cropped_signal)
    ret_starttime = starttime; ret_endtime = starttime + croptime
    return sampling_frequency, cropped_normalized_signal, filename, ret_starttime, ret_endtime

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
    if USE_DEBUGPRINT:
        print(f"f, Pxx_den  = {f} - {Pxx_den} [s]")
        print(f"f, Pxx_den  = {len(f)} - {len(Pxx_den)} ")
    return f, Pxx_den


def calculate_fft(fft_sampling_frequency, fft_source_signal):
    N = np.size(fft_source_signal)
    sample_stepping = 1.0 / fft_sampling_frequency
    xf = fftfreq(N, sample_stepping)[:N // 2]
    if USE_CUDA:
        cache = cp.fft.config.get_plan_cache()
        cache.clear()
        cp._default_memory_pool.free_all_blocks()
        if USE_DEBUGPRINT:
            print("Used GPU cache before calculation:", cp.get_default_memory_pool().used_bytes() / 1024, "kB")
        cp.cuda.runtime.memGetInfo()
        cyNorm = cp.array(fft_source_signal)
        cyf = cp.fft.fft(cyNorm)[:N // 2]
        yf = cp.asnumpy(cyf)
        # flush GPU memory to avoid cuda memory leak
        cache = cp.fft.config.get_plan_cache()
        cache.clear()
        if USE_DEBUGPRINT:
            print("Used GPU cache after calculation:", cp.get_default_memory_pool().used_bytes() / 1024, "kB")
    else:
        yfn = fft(fft_source_signal)
        yf = yfn[:N // 2]
    yf = np.abs(yf / np.max(yf)) # normalize and get amplitude density

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
    starttime = 1000.0
    timelimit = 40.0  # Limit is 400s due GPU low memory size
    all_tuples = []
    # TODO: add new files or snippets here:
    #all_tuples.append((read_sourcefile("ok2", starttime, timelimit)))
    #ret_tuple=(read_sourcefile("nok2", starttime, timelimit))
    #all_tuples.append(read_sourcefile("nok2", starttime, timelimit))
    all_tuples.append(read_sourcefile("visc3_ultrasonic", 25, timelimit))
    all_tuples.append(read_sourcefile("visc3_ultrasonic", 150, timelimit))
    #all_tuples.append(read_sourcefile("visc3_ultrasonic", 320, timelimit))
    #all_tuples.append(read_sourcefile("visc3_ultrasonic", 500, timelimit))
    all_tuples.append(read_sourcefile("visc3_ultrasonic", 700, timelimit))
    all_tuples.append(read_sourcefile("visc3_ultrasonic", 900, timelimit))
    all_tuples.append(read_sourcefile("visc3_ultrasonic", 1000, timelimit))
    all_tuples.append(read_sourcefile("visc3_ultrasonic", 1100, timelimit))

    cols = 2
    rows = len(all_tuples)

    if USE_PLOTGRAPH:
        fig, ax = plt.subplots(rows, cols, sharex=True)

    for index, one_tuple in enumerate(all_tuples):
        this_sampling_frequency = one_tuple[0]; this_signal = one_tuple[1]; this_name = one_tuple[2];

        fft_return = calculate_fft(this_sampling_frequency, this_signal)  # returns x,y values


        if USE_PLOTGRAPH:
            time = np.linspace(one_tuple[3], one_tuple[4], this_signal.shape[0])  # start, end, spacing
            ax[index, 0].title.set_text(this_name)
            ax[index, 0].plot(time,this_signal)
            ax[index, 1].plot(fft_return[0], fft_return[1])

    fig.tight_layout()
    plt.show()
