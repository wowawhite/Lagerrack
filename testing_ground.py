#!/usr/bin/env python
import numpy as np
import os.path


from scipy.io.wavfile import read

from scipy import signal
import matplotlib.pyplot as plt

rng = np.random.default_rng()



# import tensorflow as tf

# program control global variables
USE_CUDA = False
USE_PLOTGRAPH = True
USE_DEBUGPRINT = True

if (USE_DEBUGPRINT):
    print("Debug printing enabled")
    pass




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


def read_sourcefile(filename, croptime):
    work_dir = "C:\\Users\\User\\Desktop\\Testbetrieb\\snippets"
    base_filename = filename
    file_type = ".wav"
    target_file = base_filename + file_type
    wavefile = os.path.join(work_dir, target_file)
    samplingfrequency, full_signal = read(wavefile)

    sig_length = full_signal.shape[0] / samplingfrequency
    if (USE_DEBUGPRINT):
        print(f"Opening file:{wavefile}")
        print(f"sampling_frequency = {samplingfrequency}")
        print(f"shape[0] = {full_signal.shape[0]}")
        print(f"full signal length  = {sig_length} [s]")
    cropped_length, cropped_N, cropped_signal = \
        timecropping(sig_length, samplingfrequency, full_signal, croptime)
    cropped_normalized_signal = normalizing(cropped_signal)

    return samplingfrequency, cropped_normalized_signal



def calculate_spectraldensity():
    fs = 10e3
    N = 1e5
    amp = 2 * np.sqrt(2)
    freq = 1234.0
    noise_power = 0.001 * fs / 2
    time_t = np.arange(N) / fs
    x = amp * np.sin(2 * np.pi * freq * time_t)
    x += rng.normal(scale=np.sqrt(noise_power), size=time_t.shape)
    f, Pxx_den = signal.periodogram(x, fs)

    return f, Pxx_den





if __name__ == "__main__":
    timelimit = 50.0  # Limit is 400s due GPU low memory size
    sampling_frequency, signal = read_sourcefile("ok2", timelimit)
    #sampling_frequency2, signal2 = read_sourcefile("nok2", timelimit)

    #time = np.linspace(0., signal.shape[0] / sampling_frequency, signal.shape[0])  # start, end, spacing
    #time2 = np.linspace(0., signal2.shape[0] / sampling_frequency, signal2.shape[0])  # start, end, spacing


    SD_ok1,ok = calculate_spectraldensity()
    print(f"f, Pxx_den  = {SD_ok1},{ok} [s]")
