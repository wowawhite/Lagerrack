from nptdms import TdmsFile, TdmsChannel, TdmsGroup
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import subprocess
itera = 0
# mypath = "/home/wowa/Schreibtisch/Measurement07.10.2023-18_57_49.tdms"
mypath = "/home/wowa/Schreibtisch/wk_messungen/Measurement28.10.2023-10_48_23.tdms"

with TdmsFile.open(mypath) as tdms_file:
    mygroups = tdms_file.groups()

    mygroup = tdms_file['Bearing Testrack']

    # for mychannel in mygroup.channels():  # type list
    #     print(mychannel)  # type <class 'nptdms.tdms.TdmsChannel'>
    #     datastuff = mychannel[0:10]
    #     print(datastuff)

    # print(len(mygroup.channels()))
    mychannels = mygroup.channels()
    #'Laufzeit[s]'
    #'Kraft[N]'
    #'Drehzahl [U/min] '
    #'Drehmoment[mNm]'
    #'Temperatur [Grad Celsius]'

    # blop = mychannels[0]
    # print(blop[:])
    # print(mychannels[0][:])
    # print(mychannels[3][:])


    # fig, axs = plt.subplots(5, sharex=True)
    # for onechannel in mychannels:
    #     axs[itera].plot(mychannels[0][:], onechannel[:])
    #     itera = itera + 1
    # fig.tight_layout()
    # plt.show()

# tdms file closed here!
