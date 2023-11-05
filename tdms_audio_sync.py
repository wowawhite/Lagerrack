from nptdms import TdmsFile, TdmsChannel, TdmsGroup
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import soundfile as sf
import librosa as lr
import subprocess
itera = 1
#mypath = "/home/wowa/Schreibtisch/Measurement07.10.2023-18_57_49.tdms"
filename = "runup3_ultrasonic_cut.flac"
filepath = "/home/wowa/Schreibtisch/wk_messungen/"
outpath = "/home/wowa/Schreibtisch/wk_messungen/snippets/"
mypath = filepath + filename

track_seconds = range(0,17527,14)
print(len(track_seconds))
freq_match = 49
fileres = ".wav"
mysamplerate = lr.get_samplerate(path=mypath)
for snippet in track_seconds:
    outfile = outpath + str(freq_match) + fileres
# librosa not working correctly. TODO: https://pysoundfile.readthedocs.io/en/0.8.1/#module-soundfile
    snippet_signal, mysamplerate = lr.load(mypath, sr=mysamplerate, dtype='int16', offset=snippet, duration=14)
    sf.write(file=outfile, data=snippet_signal, samplerate=mysamplerate, subtype='PCM_16')
    print(f"wrote file {outfile}")

# full_signal = lr.get_duration(path=mypath)
# samplerate = lr.get_samplerate(path=mypath)
# print(f"Opening file:{mypath}")
# print(f"samplerate = {samplerate}")
# print(f"samples total = {samplerate*full_signal}")
# print(f"full signal length  = {full_signal} [s]")

# with TdmsFile.open(mypath) as tdms_file:
#     mygroups = tdms_file.groups()
#     mygroup = tdms_file['Bearing Testrack']
    # for mychannel in mygroup.channels():  # type list
    #     print(mychannel)  # type <class 'nptdms.tdms.TdmsChannel'>
    #     datastuff = mychannel[0:10]
    #     print(datastuff)
    # mychannels = mygroup.channels()
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
    # fig.tight_layout()
    # plt.show()

# tdms file closed here!
# for onechannel in mychannels:
#     axs[itera].plot(mychannels[0], onechannel[:])
#     itera = itera + 1
#