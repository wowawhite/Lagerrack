import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from pyqtgraph.dockarea import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import QtCore, QtGui, QtWidgets, uic
import numpy as np


def on_double_click_out(event):
    mouseEvent = event[0]
    mousePoint = mouseEvent.pos()
    if mouseEvent.double():
        print("Double click")
    if p.p1.sceneBoundingRect().contains(mousePoint):
        print('x=', mousePoint.x(), ' y=', mousePoint.y())


class Plotter():
    def __init__(self):
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        self.win = pg.GraphicsLayoutWidget(show=True)
        self.win.resize(1000, 500)
        self.win.setWindowTitle('pyqtgraph example: dockarea')

        self.p1 = self.win.addPlot()
        self.win.show()


p = Plotter()
proxy = pg.SignalProxy(p.win.scene().sigMouseClicked, rateLimit=60, slot=on_double_click_out)

if __name__ == '__main__':
    import sys

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtWidgets.QApplication.instance().exec_()