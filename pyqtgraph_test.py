# -*- coding: utf-8 -*-
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QApplication
import numpy as np
from threading import Thread
from queue import Queue
from numpy import arange, sin, cos, pi
from scipy.fftpack import fft, rfft
import pyqtgraph as pg
import sys
import multiprocessing

class Plot2D():
    def __init__(self):
        self.traces = dict()

        #QApplication.setGraphicsSystem('raster')
        self.app = QApplication([])
        #mw = QtGui.QMainWindow()
        #mw.resize(800,800)

        self.win = pg.GraphicsLayoutWidget(title="Basic plotting examples")
        self.win.resize(1000,600)
        self.win.setWindowTitle('pyqtgraph example: Plotting')

        # Enable antialiasing for prettier plots
        pg.setConfigOptions(antialias=True)

        self.canvas = self.win.addPlot(title="Pytelemetry")
        self.canvas.setYRange(-10, 100, padding=0)

    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QApplication.instance().exec_()

    def trace(self,name,dataset_x,dataset_y):
        if name in self.traces:
            self.traces[name].setData(dataset_x,dataset_y)
        else:
            self.traces[name] = self.canvas.plot(pen='y')

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    p = Plot2D()
    i = 0

    def signal():
        rate = 300000 # sampling rate
        t = np.arange(0, 10, 1/rate)
        sig = np.sin(2000*np.pi*4*t) + np.sin(2000*np.pi*7*t) + np.random.randn(len(t))*0.02 #4k + 7k tone + noise
        return sig

    def update():
        rate = 300000 # sampling rate
        z = 20*np.log10(np.abs(np.fft.rfft(signal()))) #rfft trims imag and leaves real values
        f = np.linspace(0, rate/2, len(z))
        p.trace("Amplitude", f, z)

    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: update())
    timer.start(10)
    p.start()
    t1 = multiprocessing.Process(target=signal)
    t1.start()
