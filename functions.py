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

import cupy as cp
# do not use  scipy.fft!

import tensorflow as tf

