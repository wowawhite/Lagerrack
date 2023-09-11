"""
from __future__ import division

import win32com.client
import win32com.client.dynamic
import win32api
import win32con
import win32com.client as win32
import win32clipboard
from win32com.client import Dispatch
from sqlite3 import dbapi2 as sqlite
import StringIO
import os
import wx
import wx.lib.dialogs
import wx.grid
import math
import sys
import numpy as np
import lvm_read
import matplotlib as mpl
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as Canvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2Wx as Toolbar
import time
import seaborn as sns # Modificare stil chart

# In the file: C:\Python24\Lib\site-packages\matplotlib\mpl-data\matplotlibrc
# file must be set the variable:        backend      : WXAgg

#Public_LCS=['Black', 'Red','Blue','Orange','Cyan','Violet','Brown','Pink', 'Green','Gray']
Public_LCS=['Black', 'SLATEGREY', 'SLATEBLUE','STEELBLUE','Black']

def ExtragTextMemory(cursor, NumeText):
		cursor.execute("SELECT fisier FROM objects where nume='"+NumeText+"'")
		blob=str(cursor.fetchone()[0])
		return blob

def ExtragImageMemory(cursor, NumeImage):
		cursor.execute("SELECT fisier FROM objects where nume='"+NumeImage+"'")
		blob=cursor.fetchone()[0]
		img = wx.ImageFromStream(StringIO.StringIO(blob))
		return img

def IsNumeric(s): # Test if a variable is numeric or not
	try:
		float(s)
	except ValueError:
		return False
	else:
		return True

def Put_Clipboard(String_Clip):  # Copy a string to Windows clipboard
		win32clipboard.OpenClipboard()
		win32clipboard.EmptyClipboard()
		win32clipboard.SetClipboardData(win32con.CF_TEXT, String_Clip) # Set clipboard data
		win32clipboard.CloseClipboard()

def Calc_Amplitude(Amplitudine, k, START, STOP, iTOT):  # Calculate signal amplitude
		cnt = n = 0
		SumColReal = SumColImag = 0. 
		for lin in range(START,STOP):
			constanta = -2 * math.pi / iTOT
			SumColReal = SumColReal + Amplitudine[cnt] * math.cos( constanta * n * k ) 
			SumColImag = SumColImag + Amplitudine[cnt] * math.sin( constanta * n * k )
			n = n + 1 ; cnt=cnt+1
		Amp = math.sqrt(SumColReal * SumColReal + SumColImag * SumColImag)
		return Amp

class Plot(wx.Panel):
	def __init__(self, parent, id = -1, dpi = None, **kwargs):
		wx.Panel.__init__(self, parent, id=id, **kwargs)
		self.figure = mpl.figure.Figure(dpi=dpi, figsize=(2,2))
		self.canvas = Canvas(self, -1, self.figure)
		self.toolbar = Toolbar(self.canvas)
		self.toolbar.Realize()
		self.toolbar.Hide()

		sizer = wx.BoxSizer(wx.VERTICAL)
		sizer.Add(self.canvas,1,wx.EXPAND)
		sizer.Add(self.toolbar, 0 , wx.LEFT | wx.EXPAND)
		self.SetSizer(sizer)

class PlotNotebook(wx.Panel):
	def __init__(self, parent, id = -1):
		wx.Panel.__init__(self, parent, id=id)
		self.nb = wx.Notebook(self)
		sizer = wx.BoxSizer()
		sizer.Add(self.nb, 1, wx.EXPAND)
		self.SetSizer(sizer)

	def add(self,name="plot"):   # Add new the page	
		page = Plot(self.nb)
		self.nb.AddPage(page,name)
		return page.figure
	
	def OnChangeSelection(self, page):   # Change the page		
		self.nb.ChangeSelection(page)

class Table_Grid(wx.Frame): # Create grid table to display numerical results
		def __init__(self,parent, id,title, position,size, sirgrid):
			wx.Frame.__init__(self, parent, id,title,position, (500,590), style= 
				wx.DEFAULT_FRAME_STYLE ^ (wx.RESIZE_BORDER | wx.MINIMIZE_BOX | wx.MAXIMIZE_BOX ))
			panel = wx.Panel(self)
			if title=="Input Data" or title=="DFT Amplitude":
				Grid = wx.grid.Grid(panel, pos=(100,0), size=(300, 500))
				Grid.CreateGrid(0,3)
				Grid.SetColLabelValue(0,"ID")
				name="Time" if title=="Input Data" else "Frequency"
				Grid.SetColLabelValue(1,name)
				Grid.SetColLabelValue(2,"Amplitude")				
			if title=="Overlapped DFT" or title=="Points Selected":
				Grid = wx.grid.Grid(panel, pos=(30,0), size=(460, 500))
				Grid.CreateGrid(0,5)
				Grid.SetColLabelValue(0,"ID")
				Grid.SetColLabelValue(1,"Time total")
				Grid.SetColLabelValue(2,"Frequency")
				Grid.SetColLabelValue(3,"Amplitude")
				Grid.SetColLabelValue(4,"No.cycles")
				Grid.SetColSize(4,70)
			Grid.SetRowLabelSize(50)
			Grid.SetColSize(0,50)			
			Grid.EnableEditing(0)
			Grid.SetDefaultCellAlignment(wx.ALIGN_CENTRE,wx.ALIGN_CENTRE)

			lgrid=0		
			SG=sirgrid.split("\n")
			self.SetTitle(title+" - "+str(len(SG)-1)+" records")
			for x in SG:				
				if len(x)>0:
					lin=x.split("\t")
					Grid.AppendRows(1)
					Grid.SetCellValue(lgrid, 0, lin[0])
					Grid.SetCellValue(lgrid, 1, lin[1])
					Grid.SetCellValue(lgrid, 2, lin[2])				
					if title=="Overlapped DFT" or title=="Points Selected":
						Grid.SetCellValue(lgrid, 3, lin[3])
						Grid.SetCellValue(lgrid, 4, lin[4])
					lgrid+=1	

			Btn_OK= wx.Button(panel, -1, "OK", (200,520), (100,-1))
			self.Bind(wx.EVT_BUTTON, self.OnClose,Btn_OK)
			
			self.CenterOnScreen()
			self.Show()
		
		def OnClose(self, event):
			self.Close(True)
			self.Destroy()

# ================================================================
class Calcul_DFT(wx.Frame):

	def __init__(self):
		displaySize= wx.DisplaySize()
		SirTitlu="PyFEST V .1.0.0.2019/2020 - Discrete Fourier Transformation"
		SirTitlu=SirTitlu + " - " + "@ Octomber 2019 - March 2020 - D. Nedelcu & G.R. Gillich - 'Eftimie Murgu' University of Resita, Romania"
		wx.Frame.__init__(self, None, -1, title=SirTitlu, size=(displaySize[0]*0.95, displaySize[1]*0.95))
		
		mpl.rc('xtick', labelsize=20) 
		mpl.rc('ytick', labelsize=20)		
		
		self.statusbar = self.CreateStatusBar()  # Define statusBar
		self.statusbar.SetFieldsCount(4)
		self.statusbar.SetStatusWidths([-70, -20, -40, -10])
		self.statusbar.SetStatusText("@ Octomber 2019 - March 2020 - D. Nedelcu & G.R. Gillich - 'Eftimie Murgu' University of Resita, Romania",1)
		
		self.plotter = PlotNotebook(self)  # Create Notebook
		mpl.rcParams['legend.fontsize'] = 14
		self.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.OnPageChanged)

		# Create 4 charts (axis) area: "Input_Data", "DFT_Amplitude", "Overlapped_DFT", "Points_Selected"
		self.axa1 = self.plotter.add("Input_Data").gca()

		self.axa2 = self.plotter.add("DFT_Amplitude").gca()
		self.axa2.set_title("Maximal Peaks", fontsize=16, fontweight='bold')
		self.axa2.grid(True)
		self.axa2.set_xlabel("Frequency [Hz]", fontsize=20, fontweight='bold')
		self.axa2.set_ylabel("DFT  Amplitude", fontsize=20, fontweight='bold')
		self.axa2.figure.canvas.mpl_connect('motion_notify_event', self.MouseMotion)
		self.axa2.figure.canvas.mpl_connect('button_press_event', self.OnPick_DFT_Amplitude)
						
		self.axa3 = self.plotter.add("Overlapped_DFT").gca()
		self.axa3.set_title("Overlapped DFT for  the cropped signals", fontsize=16, fontweight='bold')
		self.axa3.grid(True)
		self.axa3.set_xlabel("Frequency [Hz]", fontsize=20, fontweight='bold')
		self.axa3.set_ylabel("DFT  Amplitude", fontsize=20, fontweight='bold')
		self.axa3.figure.canvas.mpl_connect('motion_notify_event', self.MouseMotion)
		self.axa3.figure.canvas.mpl_connect('button_press_event', self.OnPick_Overlapped_DFT)
		
		self.axa4 = self.plotter.add("Points_Selected").gca()
		self.axa4.set_title("Values for number of cycles selected", fontsize=16, fontweight='bold')
		self.axa4.grid(True)
		self.axa4.set_xlabel("Frequency [Hz]", fontsize=20, fontweight='bold')
		self.axa4.set_ylabel("PSD Amplitude ", fontsize=20, fontweight='bold')
		self.axa4.figure.canvas.mpl_connect('motion_notify_event', self.MouseMotion)		

		# Verify existence of RESULTS folder and create if not exist
		dir_RESULTS=os.getcwd()+'\RESULTS'
		if os.path.exists(dir_RESULTS)==False:
				os.mkdir(dir_RESULTS)    
		
		# Define progress bar
		self.Pro_Bar = wx.Gauge(self.statusbar, -1, style=wx.GA_HORIZONTAL|wx.GA_SMOOTH)
		rect = self.statusbar.GetFieldRect(2)
		self.Pro_Bar.SetPosition((rect.x+1, rect.y+1)) ; self.Pro_Bar.SetSize((rect.width-2, rect.height-2))
		
		self.AxaActiva=1        # Set current axis
		self.NumeFigura=[]	
		self._TOOLBAR()         # Create toolbar
		self.CenterOnScreen()
		self.Show()
		self.Calculation0()     # Call local function
	
	def RecretareAxa1(self):
		self.axa1.set_title("Input Data", fontsize=20, fontweight='bold')
		self.axa1.grid(True)
		self.axa1.set_xlabel("Time [s]", fontsize=24, fontweight='bold')
		self.axa1.set_ylabel("Amplitude  [mm/s2]", fontsize=24, fontweight='bold')
		self.axa1.figure.canvas.mpl_connect('motion_notify_event', self.MouseMotion)
		self.axa1.legend(loc='upper right', prop={'size': 9})
		self.axa1.figure.canvas.draw()
	
	def Calculation0(self):

		self.RecretareAxa1()

		# Select Excel-CSV or LabView-LVM file with input data
		wildcard = "Excel CSV / LabView LVM (*.csv; *.lvm)|*.csv;*.lvm"
		titlu_open="Open an Excel CSV / LabView LVM file to calculate Discrete Fourier Transformation (DFT)"		
		dlg = wx.FileDialog(self, titlu_open, os.getcwd(), "", wildcard, wx.OPEN | wx.MULTIPLE)
		if dlg.ShowModal() == wx.ID_CANCEL:
			dlg.Destroy()
			self.Close(True)
			self.Destroy()			
			return
		FILENAME = dlg.GetFilename() ; self.OpenFile=FILENAME
		dir_file = dlg.GetDirectory()
		dlg.Destroy()
		PozPoint=FILENAME.rfind(".")
		extensie=FILENAME[PozPoint+1:len(FILENAME)] 
		self.NameFileCSV=self.NameImg = FILENAME[0:PozPoint]
		
		cnt=0 ; Time=[] ; Amplitudine=[]				
		f = open(dir_file+"/"+FILENAME, "r")			
		if extensie.upper()=="CSV":		
			for x in f:   # Read Excel CSV file
				if x.rfind(",")==-1: #Read number of CSV lines
					Ntotal=int(x) 
				else:
					lin=x.split(",") #Read lines and split contents by comma
					Time.append(float(lin[0])) ; Amplitudine.append(float(lin[1]))
		else:			# Read LabView LVM file
			lvm = lvm_read.read(dir_file+"\\"+FILENAME, read_from_pickle=False)
			Date_lvm=lvm[0]['data']	
			timp=[] ; amplitudine=[]
			for lin in Date_lvm:
				Time.append(float(format(lin[0], '.8f')))
				Amplitudine.append(float(format(lin[1], '.8f')))
			Ntotal=len(Time)
		f.close()		
		self.SetTitle("PyFEST V .1.0.0.2019 - Discrete Fourier Transformation ; File: "+dir_file+"/"+FILENAME)
				
		self.Ttotal=Ttotal=Ttotal_0=max(Time)
		self.Ntotal=Ntotal_0=iTotal =Ntotal		
		self.Time=Time ; self.Amplitudine=Amplitudine
		
		# Save Input Data points to CSV file
		file_CSV = open(os.getcwd()+"/RESULTS/"+self.NameFileCSV+"-"+"Input_Data.CSV",'w')
		file_CSV.write("Point, Time, Amplitude"+'\n')
		for i in range(0,len(Time)):	
			sircsv=str(i+1)+","+str(Time[i])+","+ str(Amplitudine[i])+"\n"
			file_CSV.write(sircsv)
		file_CSV.close		
		
		self.IDP="" 	# Memorize Input Data points to save in Excel file & screen table 
		for i in range(0,len(Time)):
			self.IDP=self.IDP+"%4d" % (i+1)+"\t"+"%10.4f" % Time[i]+"\t"+"%10.4f" % Amplitudine[i]+"\n"
		
		# Chart with "Input data" points
		titlu_axa1=str(len(Time))+ " records" + " from file: '"+FILENAME+"' with Time maxim="+str(Time[len(Time)-1])+ " sec"
		self.axa1.set_title(titlu_axa1, fontsize=20, fontweight='bold')		
		self.axa1.plot(Time, Amplitudine, '.-', markersize=7, picker=5, color="Blue", linewidth=5.0, label = "Initial signal")
		self.axa1.figure.canvas.draw()
		self.AxaActiva=1	
		
		OLD_MinTime=min(self.Time)
		OLD_MaxTime=max(self.Time)
			
		# Keyboard input of the minimal time	
		tit = "Input MINIMAL time for truncate signal" ; msg = "Enter a value "
		dlg = wx.TextEntryDialog(self, msg, tit, str(min(self.Time)), style = wx.OK | wx.CANCEL)
		if dlg.ShowModal() == wx.ID_CANCEL:
			dlg.Destroy()
			self.Close(True)
			self.Destroy()			
			return		
		MinTime=float(str(dlg.GetValue()))	
		dlg.Destroy()

		# Keyboard input of the maximal time	
		tit = "Input MAXIMAL time for truncate signal" ; msg = "Enter a value "
		dlg = wx.TextEntryDialog(self, msg, tit, str(max(self.Time)), style = wx.OK | wx.CANCEL)
		if dlg.ShowModal() == wx.ID_CANCEL:
			dlg.Destroy()
			self.Close(True)
			self.Destroy()			
			return		
		MaxTime=float(str(dlg.GetValue()))	
		dlg.Destroy()
		
		if  MinTime<>OLD_MinTime or MaxTime<>OLD_MaxTime:  # Trunchiere semnal
				newTime = [] ; newAmplitudine = [] ; newTimeGraph = []
				for i, ( tmp, amp) in enumerate(zip(self.Time,self.Amplitudine )):
					if (tmp >= MinTime) and (tmp <= MaxTime):
						newTimeGraph.append(tmp)
						newTime.append(tmp-MinTime)
						newAmplitudine.append(amp)
				self.Time = newTime
				self.Amplitudine = newAmplitudine
				self.Time = [] ; self.Amplitudine = []
				for i, ( tmp, amp) in enumerate(zip(newTime,newAmplitudine)):
					self.Time.append(tmp) ; self.Amplitudine.append(amp)

				titlu_axa1=str(len(self.Time))+ " records" + " from file: '"+FILENAME+"' with Time maxim="+str(self.Time[len(self.Time)-1])+ " sec"
				self.axa1.set_title(titlu_axa1, fontsize=20, fontweight='bold')		
				self.axa1.plot(newTimeGraph, self.Amplitudine, '.-', markersize=7, picker=5, color="Red", linewidth=5.0, label = "Truncated signal")
				self.axa1.figure.canvas.draw()

				self.Ttotal=max(self.Time)
				Ttotal=max(self.Time)
				Ttotal_0=max(self.Time)

				self.Ntotal=len(self.Time)
				Ntotal=len(self.Time)
				Ntotal_0=len(self.Time)	
				iTotal =len(self.Time)		

				# Save Input Data points to CSV file
				file_CSV = open(os.getcwd()+"/RESULTS/"+self.NameFileCSV+"-"+"Input_Data_Truncated.CSV",'w')
				file_CSV.write("Point, Time, Amplitude"+'\n')
				for i in range(0,len(self.Time)):	
					sircsv=str(i+1)+","+str(self.Time[i])+","+ str(self.Amplitudine[i])+"\n"
					file_CSV.write(sircsv)
				file_CSV.close		
				
				self.IDP="" 	# Memorize Input Data points to save in Excel file & screen table 
				for i in range(0,len(self.Time)):
					self.IDP=self.IDP+"%4d" % (i+1)+"\t"+"%10.4f" % self.Time[i]+"\t"+"%10.4f" % self.Amplitudine[i]+"\n"
			
		# Keyboard input of the maximal expected frequency		
		tit = "Input maximal expected frequency" ; msg = "Enter an integer number "
		dlg = wx.TextEntryDialog(self, msg, tit, "", style = wx.OK | wx.CANCEL)
		if dlg.ShowModal() == wx.ID_CANCEL:
			dlg.Destroy()
			self.Close(True)
			self.Destroy()			
			return

		MaxExpFreq=int(str(dlg.GetValue()))	
		dlg.Destroy()

		# Calcul DFT and save DFT_Amplitude points in CSV file
		self.Pro_Bar.SetRange(int(Ntotal/2)+1) ; self.Pro_Bar.Show() 
		path_file_CSV=os.getcwd()+"/RESULTS/"+self.NameFileCSV+"-"+"DFT_Amplitude.CSV"
		file_CSV = open(path_file_CSV,'w')
		file_CSV.write("Point, Frequency, Amplitude"+'\n')
		kamp_grf=[] ; Vamp_grf=[] ; iii=0 
		NrIter=MaxExpFreq/self.Ttotal-1
		for k in range(0,int(self.Ntotal/2)+1):
			self.Pro_Bar.SetValue(iii) ; self.statusbar.SetStatusText(str(iii),3) ; iii+=1
			Amp=Calc_Amplitude(self.Amplitudine, k, 1, iTotal+1, iTotal)
			frecy=float(k) / float(Ttotal_0)
			kamp_grf.append(frecy)
			Vamp_grf.append(Amp)			
			sircsv=str(iii)+","+str(frecy)+","+ str(Amp)+"\n"
			file_CSV.write(sircsv)
			if frecy>MaxExpFreq:
				break	
		file_CSV.close
		self.Pro_Bar.Hide() ; self.statusbar.SetStatusText("",3)

		# Chart with "DFT_Amplitude" points - axa2
		self.axa2.plot(kamp_grf, Vamp_grf, '.-', markersize=9, picker=5, color="Blue", linewidth=5.0)
		
		self.DFTA=""   # Memorize DFT_Amplitude points to save in Excel file & screen table
		for i in range(0,len(kamp_grf)):
			self.DFTA=self.DFTA+"%4d" % (i+1)+"\t"+"%30.20f" % kamp_grf[i]+"\t"+"%30.20f" % Vamp_grf[i]+"\n"			
			
		# Highlight maximal points in self.axa2 -> DFT_Amplitude and memorize them in "Lista_maxime"
		icol=0
		self.Lista_maxime=[]
		for i in range(1,len(Vamp_grf)-1):
			if (Vamp_grf[i] > Vamp_grf[i - 1]) and (Vamp_grf[i] > Vamp_grf[i + 1]):
				culoare=Public_LCS[icol]
				self.axa2.plot(kamp_grf[i-1], Vamp_grf[i-1], 'o', markersize=8, picker=5, color=culoare)	
				self.axa2.plot(kamp_grf[i], Vamp_grf[i], 'o', markersize=8, picker=5, color=culoare)
				self.axa2.plot(kamp_grf[i+1], Vamp_grf[i+1], 'o', markersize=8, picker=5, color=culoare)	
				sirtxt="   Freq / Ampl=" + str('%.4f' % kamp_grf[i]) + " / " +  str('%.2f' % Vamp_grf[i])+ \
						" / " +  str('%.4f' % (Vamp_grf[i]*2/self.Ntotal))
				self.axa2.text(kamp_grf[i], Vamp_grf[i], sirtxt, ha='left', va= 'top',color=culoare, fontsize=20, fontweight='bold')
				Lst_linie=[]
				Lst_linie.extend([kamp_grf[i-1], kamp_grf[i], kamp_grf[i+1], Vamp_grf[i-1], Vamp_grf[i], Vamp_grf[i+1] ])
				self.Lista_maxime.append(Lst_linie)
				self.statusbar.SetStatusText(str(icol),3)
				icol+=1
				if icol>len(Public_LCS)-1:  icol=0
		title_axa2=str(len(self.Lista_maxime))+" Maximal Peaks found for maximal expected frequency = "+str(MaxExpFreq)
		self.axa2.set_title(title_axa2, fontsize=20, fontweight='bold')
		self.plotter.OnChangeSelection(1)
		self.AxaActiva=2
		sir="Select maximal frequency by right button mouse...."
		self.statusbar.SetStatusText(sir, 0)

	def Calculation1(self, FrecSel):
		self.statusbar.SetStatusText("",0)		
		PB_cnt = 0
		iTotal = self.Ntotal		
		self.Vct_iTotal=[] ; self.Vct_Ttotal=[] ; self.Vct_frec=[] ; self.Vct_MaxAmp=[] ; self.Vct_NoCycles=[]
		Kreferinta=int(FrecSel * self.Ttotal)

		self.Pro_Bar.SetRange(self.Ntotal) ;  self.Pro_Bar.Show() 
		
		path_file_CSV1=os.getcwd()+"/RESULTS/"+self.NameFileCSV+"-"+"Overlapped_DFT.CSV"
		file_CSV1 = open(path_file_CSV1,'w')
		sircsv="Point, Signal_length, Frequency, Amplitude, No.cycles"+'\n'
		file_CSV1.write(sircsv)	
		
		TimeSTART=self.Ttotal ; Tend=TimeSTART-2./FrecSel
		NrMaxime=10
		while True:
			Xslice=self.Time[0:iTotal] 
			if len(Xslice)==0:
				break			
			Ttotal = max(Xslice)			
			iamp = 0 ; kamp=[] ; Vamp=[]
			for k in range(Kreferinta-6, Kreferinta+2):
				PB_cnt+=1
				self.Pro_Bar.SetValue(PB_cnt)
				self.statusbar.SetStatusText(str(PB_cnt),3)
				Amp=Calc_Amplitude(self.Amplitudine, k, 1, iTotal+1, iTotal)
		
				# Memorize array for multiple maximus calculation
				iamp = iamp + 1
				kamp.append(float(k) / float(Ttotal))
				Vamp.append(Amp)
				
			# Multiple maximus calculation
			for kkk in range(1, NrMaxime+1):
				MaxAmp = -1
				for jj in range(1,iamp):
					if (Vamp[jj] > MaxAmp) and (Vamp[jj] <> 12345):
						MaxAmp = Vamp[jj] 
						frec = kamp[jj] 
						Vamp[jj] = 12345
				if MaxAmp<>-1:
					self.Vct_iTotal.append(iTotal)		# Number of samples in signal N [-]
					self.Vct_Ttotal.append(Ttotal)		# Signal length in time t [s]					
					self.Vct_frec.append(frec)  		# Frequencies calculated for different signal lengths [Hz]
					self.Vct_MaxAmp.append(MaxAmp)  	# Amplitudes now calculated by the application
					self.Vct_NoCycles.append(int(Ttotal * frec+0.5)) # Number of cycles
					sircsv=str(iTotal)+","+ str(Ttotal)+","+ str(frec)+","+str(MaxAmp)+","+str(Ttotal * frec)+"\n"
					file_CSV1.write(sircsv)		
			iTotal = iTotal - 2			
			if (Ttotal < Tend) or (Ttotal<0):   # End ciclu WHILE	
				break
		self.Pro_Bar.Hide() ; self.statusbar.SetStatusText("",3)
		file_CSV1.close()
		
		self.OFTA=""    # Memorize Overlapped_DFT points to save in Excel file & screen table  
		for i in range(0,len(self.Vct_frec)):
			self.OFTA=self.OFTA+"%4d" % self.Vct_iTotal[i]+"\t"+"%10.4f" % self.Vct_Ttotal[i]+"\t"+ \
					"%10.4f" % self.Vct_frec[i]+"\t"+"%10.4f" % self.Vct_MaxAmp[i]+"\t"+"%4d" % self.Vct_NoCycles[i]+"\n"			
		
		# Create list with unique values of number of cycles 
		NoCyclesUnique=[]
		for x in self.Vct_NoCycles:
			if x not in NoCyclesUnique:
				NoCyclesUnique.append(x)
		NoCyclesUnique.sort() ; NoCyclesUnique.reverse()
		
		# Chart with "Overlapped" data points in self.axa 3 - Overlapped
		icol=0
		for NrCycles in  NoCyclesUnique:
			self.Ax3_frec=[] ; self.Ax3_MaxAmp=[] ; MxAmp=-1
			for i, ( Ax3_FREC, Ax3_AMPL, Ax3_NC) in enumerate(zip(self.Vct_frec, self.Vct_MaxAmp, self.Vct_NoCycles)):
				if NrCycles==Ax3_NC:
					if Ax3_AMPL>MxAmp:
						MxAmp=Ax3_AMPL ; MxFrec=Ax3_FREC
					self.Ax3_frec.append(Ax3_FREC)  	# Frequencies calculated for different signal lengths [Hz]
					self.Ax3_MaxAmp.append(Ax3_AMPL)  	# Amplitudes now calculated by the application
			culoare=Public_LCS[icol]
			self.axa3.plot(self.Ax3_frec, self.Ax3_MaxAmp, '-', markersize=4, picker=5, color=culoare, label="Cycle no. "+str(NrCycles), linewidth=5.0)
			sirPoint="   Freq / Ampl / Cycle no.=" + str('%.2f' % MxFrec) + " / " +  str('%.2f' % MxAmp)+ " / " +  str('%3d' % NrCycles)
			self.axa3.text(MxFrec, MxAmp, sirPoint, ha='left', va= 'bottom',color=culoare, fontsize=20, fontweight='bold')
			self.axa3.plot(MxFrec, MxAmp, 'o', markersize=8, picker=5, color=culoare)
			icol+=1
			if icol>len(Public_LCS)-1:  icol=0

		self.axa3.legend(loc='upper right')
		self.axa3.legend()
		self.plotter.OnChangeSelection(2)
		self.AxaActiva=3
		sir="Select targeted cycle by right button mouse..."
		self.statusbar.SetStatusText(sir, 0)
		
	def Calculation2(self, ID_Ciclu_Selectat):
		
		if ID_Ciclu_Selectat<4:
			wx.MessageBox("No. of cycles must be greather then 3.", "ERROR", wx.OK | wx.ICON_ERROR)
			sys.exit()
		self.statusbar.SetStatusText("",0)
		# Copy values only for selected cycle (ID_Ciclu_Selectat)
		Cic_iTotal=[] ; Cic_Ttotal=[] ; Cic_frec=[] ; Cic_MaxAmp=[] ; Cic_Ttotalxfrec=[]	
		gasit_Cicluri=0	; gasit = 0	
		for i, (itot, Ntot, frec, ampl, nrcic) in \
				enumerate(zip(self.Vct_iTotal,self.Vct_Ttotal,self.Vct_frec, self.Vct_MaxAmp, self.Vct_NoCycles)):
			frecRef = frec
			NrCicluComparatie = int(float(nrcic)+0.5)
			if (ID_Ciclu_Selectat == NrCicluComparatie):
				gasit_Cicluri=1	
				Cic_iTotal.append(itot)
				Cic_Ttotal.append(Ntot)
				Cic_frec.append(frecRef)
				Cic_MaxAmp.append(ampl**2/Ntot)
				Cic_Ttotalxfrec.append(NrCicluComparatie)

		# Sort values
		if gasit_Cicluri==1 and len(Cic_MaxAmp)>=3:
				sss=sorted(zip(Cic_iTotal, Cic_Ttotal, Cic_frec, Cic_MaxAmp, Cic_Ttotalxfrec))
				self.SRT_iTotal, self.SRT_Ttotal, self.SRT_frec, self.SRT_MaxAmp, self.SRT_NoCycles = map(list, zip(*sss))
				self.SRT_iTotal.reverse()
				self.SRT_Ttotal.reverse()
				self.SRT_frec.reverse()
				self.SRT_MaxAmp.reverse()
				self.SRT_NoCycles.reverse()
				path_file_CSV2=os.getcwd()+"/RESULTS/"+self.NameFileCSV+"-"+"Points_Selected.CSV"
				file_CSV2 = open(path_file_CSV2,'w')
				sircsv="i, Ttot, Frequency, Amplitude, No.cycles"+'\n'
				file_CSV2.write(sircsv)
				for i, (itot, Ntot, frec, ampl, nrcic) in enumerate(zip(self.SRT_iTotal,self.SRT_Ttotal,self.SRT_frec, self.SRT_MaxAmp, self.SRT_NoCycles)):
					sircsv=str(itot)+","+str(Ntot)+","+str(frec)+","+str(ampl)+","+str(nrcic)+'\n'
					file_CSV2.write(sircsv)
				file_CSV2.close()

				self.POSE=""   # Memorize Points selected points to save in Excel file & screen table 
				for i in range(0,len(self.SRT_frec)):
					self.POSE=self.POSE+"%4d" % self.SRT_iTotal[i]+"\t"+"%10.4f" % self.SRT_Ttotal[i]+ \
									 "\t"+ "%10.4f" % self.SRT_frec[i]+"\t"+"%10.4f" % self.SRT_MaxAmp[i]+ \
									 "\t"+"%4d" % self.SRT_NoCycles[i]+"\n"                   
				
				# Chart with "Points selected" data points in axa 4
				MXAMP=max(self.SRT_MaxAmp)
				AmpGraf=[]
				for x in self.SRT_MaxAmp:
					AmpGraf.append(x/MXAMP)
					
				self.axa4.plot(self.SRT_frec, AmpGraf, '.-', markersize=4, picker=5, color="Black", linewidth=5.0)	
				
				self.POSE=""   # Memorize Points selected points to save in Excel file & screen table 
				for i in range(0,len(self.SRT_frec)):
					self.POSE=self.POSE+"%4d" % self.SRT_iTotal[i]+"\t"+"%10.4f" % self.SRT_Ttotal[i]+"\t"+ \
							"%10.4f" % self.SRT_frec[i]+"\t"+"%10.4f" % self.SRT_MaxAmp[i]+"\t"+"%4d" % self.SRT_NoCycles[i]+"\n"					
					
				maxY = -1e7
				for i in range(1,len(self.SRT_MaxAmp)):
					if i<len(self.SRT_MaxAmp)-1: 
						if (self.SRT_MaxAmp[i] > self.SRT_MaxAmp[i - 1]) and (self.SRT_MaxAmp[i] > self.SRT_MaxAmp[i + 1]):
							maxY = self.SRT_MaxAmp[i]
							Ndivide = float(self.SRT_iTotal[i])
							gasit = 1
							break		

				if gasit==1:
						# Calculation of the coefficients for equation 
						X0=self.SRT_frec[i-1] ; X1=self.SRT_frec[i] ; X2=self.SRT_frec[i+1]
						Y0=self.SRT_MaxAmp[i-1] ; Y1=self.SRT_MaxAmp[i] ; Y2=self.SRT_MaxAmp[i+1]
						X02=X0*X0 ; X12=X1*X1 ; X22=X2*X2
						A = np.array([[X02, X0, 1], [X12, X1, 1], [X22, X2, 1]])
						b = np.array([Y0, Y1, Y2])
						x = np.linalg.solve(A, b) # Solving system of equations
						aa=x[0] ; bb=x[1] ; cc=x[2]

						# Calculation the maximum amplitude point
						Delta = bb * bb - 4 * aa * cc
						Xparmax = -bb / 2 / aa		;  self.POSE_Freq=Xparmax
						yparmax = -Delta / 4 / aa	;  self.POSE_Amp0=yparmax
						
						FR=( self.Ntotal-1)/self.Ttotal
						
						fex1=os.getcwd()+"/RESULTS/Constants.txt"
						connection = sqlite.connect(os.getcwd()+'\Config.db') ; cursor = connection.cursor()
						blob=ExtragTextMemory(cursor, 'Constants.txt')
						cursor.close() ; connection.close()
						open(fex1, 'wb').write(blob)						

						# Read constants from "Constants.txt" file
						CICLURI = [] ; CC= []
						with open(fex1, 'r') as f:
						    for line in f.readlines():
						        cclr,valoare = line.strip().split(',')
						        CICLURI.append((int(cclr)))
						        CC.append((float(valoare)))
						NrCritic=CICLURI[len(CICLURI)-1]
						os.remove(fex1)
						
						poz=0
						for CICLU in CICLURI:
							if CICLU==ID_Ciclu_Selectat:
								CCMEM=CC[poz]
								break    
							poz=poz+1
						if ID_Ciclu_Selectat>=NrCritic:  CCMEM=CC[len(CC)-1]
						
						# Calculation of Absolute Maximus with correction coefficients
						AAAA=1000/FR*CCMEM
						BBBB=-1
						CCCC=Xparmax
						Fcorect=(-BBBB-math.sqrt(BBBB**2-4*AAAA*CCCC))/2/AAAA
						self.POSE_Freq = Fcorect
						
						AmplCORECT=yparmax * 4. / Ndivide / FR ;  self.POSE_Amp = AmplCORECT						
						sirPoint="   Freq / Ampl=" + str('%.4f' % Fcorect) + " / " +  str('%.4f' % AmplCORECT)
						self.axa4.plot(Xparmax, yparmax/MXAMP, 'o', markersize=8, picker=5, color="Blue")						
						self.axa4.text(Xparmax, yparmax/MXAMP, sirPoint, ha='left', va= 'bottom',color='b', fontsize=20, fontweight='bold')						
						self.plotter.OnChangeSelection(3)
						self.AxaActiva=4
												
						siraxa="Points for "+str(ID_Ciclu_Selectat)+" cycles selected:"
						siraxa=siraxa+" Frequency =" + str('%.4f' % Fcorect)   
						siraxa=siraxa+" Amplitude = " + str('%.4f' % AmplCORECT) + "\n"
						self.axa4.set_title(siraxa, fontsize=20, fontweight='bold')
						
						siraxa1="Points for "+str(ID_Ciclu_Selectat)+" cycles selected:"
						siraxa1=siraxa1+"\nFrequency =" + str('%.4f' % Fcorect)   
						siraxa1=siraxa1+"\nAmplitude = " + str('%.4f' % AmplCORECT) + "\n"
						
						msjCLIP=siraxa1+"\n\nValues copied to Windows clipboard."
						Put_Clipboard(msjCLIP)
						wx.MessageBox(msjCLIP, "Info", wx.OK | wx.ICON_INFORMATION)
						
		if gasit_Cicluri==0  or gasit==0:
			sir = "The maximum cannot be determined."+"\n"+"For No. of cycles="+str(ID_Ciclu_Selectat)+"\n"
			sir=sir+"error="+str(gasit)+" / "+str(gasit_Cicluri)+"\n"
			wx.MessageBox(sir, "Discrete Fourier Transformation", wx.OK | wx.ICON_ERROR)

	def Home(self,event):
		self.axa1.figure.canvas.toolbar.home()
		self.axa2.figure.canvas.toolbar.home()
		self.axa3.figure.canvas.toolbar.home()
		self.axa4.figure.canvas.toolbar.home()

	def Pan(self,event):
		self.axa1.figure.canvas.toolbar.pan()
		self.axa2.figure.canvas.toolbar.pan()
		self.axa3.figure.canvas.toolbar.pan()
		self.axa4.figure.canvas.toolbar.pan()
		
	def Zoom(self,event):
		self.axa1.figure.canvas.toolbar.zoom()
		self.axa2.figure.canvas.toolbar.zoom()
		self.axa3.figure.canvas.toolbar.zoom()
		self.axa4.figure.canvas.toolbar.zoom()

	def OnPick_DFT_Amplitude(self, event):   # Right click mouse on "DFT_Amplitude" chart	
		Btn_MOUSE=event.button  ;  X_MOUSE=event.xdata  ;  Y_MOUSE=event.ydata	
		if Btn_MOUSE==3 and self.AxaActiva==2:  			
			dist_ant=+1e7
			for lin in self.Lista_maxime:		
				Xmax=lin[1]
				Ymax=lin[4]				
				dist_crt=math.sqrt((Xmax-X_MOUSE)*(Xmax-X_MOUSE)+(Ymax-Y_MOUSE)*(Ymax-Y_MOUSE))
				if dist_crt<dist_ant:
					dist_ant=dist_crt
					FrecSelected=Xmax
			self.Calculation1(FrecSelected)

	def OnPick_Overlapped_DFT(self, event):    # Right click mouse on "Overlapped_DFT" chart
		Btn_MOUSE=event.button  ;  X_MOUSE=event.xdata  ;  Y_MOUSE=event.ydata	
		if Btn_MOUSE==3 and self.AxaActiva==3:  		
			dist_ant=+1e7
			for i, (Frec, Ampl, NC) in enumerate(zip(self.Vct_frec, self.Vct_MaxAmp, self.Vct_NoCycles)):					
				dist_crt=math.sqrt((Frec-X_MOUSE)*(Frec-X_MOUSE)+(Ampl-Y_MOUSE)*(Ampl-Y_MOUSE))
				if dist_crt<dist_ant:
					dist_ant=dist_crt
					ID_Ciclu_Selectat=NC
			self.Calculation2(ID_Ciclu_Selectat)
			
	def MouseMotion(self, event):
		x,y = event.xdata, event.ydata
		if self.AxaActiva==1: a1,b1=self.axa1.transData.inverted().transform([event.x, event.y])
		if self.AxaActiva==2: a1,b1=self.axa2.transData.inverted().transform([event.x, event.y])
		if self.AxaActiva==3: a1,b1=self.axa3.transData.inverted().transform([event.x, event.y])
		if self.AxaActiva==4: a1,b1=self.axa4.transData.inverted().transform([event.x, event.y])
		if a1<>None and b1<>None and x<>None and y<>None:
			sir="X="+'%0.4f' % a1+",  Y="+'%0.4f' % b1 
			self.statusbar.SetStatusText(sir,1)

	def OnTabele(self, event):
		if self.AxaActiva==1:
			frm_Grid_Class1=Table_Grid(None, -1,"Input Data", wx.DefaultPosition, (720, 560), self.IDP)
		if self.AxaActiva==2:
			frm_Grid_Class2=Table_Grid(None, -1,"DFT Amplitude", wx.DefaultPosition, (720, 560), self.DFTA)
		if self.AxaActiva==3:
			frm_Grid_Class3=Table_Grid(None, -1,"Overlapped DFT", wx.DefaultPosition, (720, 560), self.OFTA)
		if self.AxaActiva==4:
			frm_Grid_Class4=Table_Grid(None, -1,"Points Selected", wx.DefaultPosition, (720, 560), self.POSE)
			
	def OnPageChanged(self, event):  # Define the current axis when Notebook page is changed
		old = event.GetOldSelection()
		new = event.GetSelection()
		if int(new)==0:  self.AxaActiva=1
		if int(new)==1:  self.AxaActiva=2
		if int(new)==2:  self.AxaActiva=3
		if int(new)==3:  self.AxaActiva=4
		event.Skip()	
		
	def OnSaveImg(self):  # Save charts as images in RESULTS folder

		File_Img1=os.getcwd()+"/RESULTS/"+self.NameImg+'_Input_Data.jpg'
		File_Img2=os.getcwd()+"/RESULTS/"+self.NameImg+'_DFT_Amplitude.jpg'
		File_Img3=os.getcwd()+"/RESULTS/"+self.NameImg+'_Overlapped_DFT.jpg'
		File_Img4=os.getcwd()+"/RESULTS/"+self.NameImg+'_Points_Selected.jpg'
		self.NumeFigura=[]
		self.NumeFigura.extend([str(File_Img1), str(File_Img2), str(File_Img3), str(File_Img4)])
		
		self.axa1.figure.savefig(File_Img1)
		self.axa2.figure.savefig(File_Img2)
		self.axa3.figure.savefig(File_Img3)
		self.axa4.figure.savefig(File_Img4)


	def OnExcel(self, event): # Export results to Excel file
		
		self.Pro_Bar.SetRange(7) ; self.Pro_Bar.Show() 
		
		Lst_Sheets= ["Input_Data", "DFT_Amplitude", "Overlapped_DFT", "Points_Selected"]
		Lst_data = [self.IDP, self.DFTA, self.OFTA, self.POSE]

		fex1=os.getcwd()+"/RESULTS/"+self.NameFileCSV+"-"+"DFT_Results.xls"
		connection = sqlite.connect(os.getcwd()+'\Config.db') ; cursor = connection.cursor()
		blob=ExtragTextMemory(cursor, 'Template_Results.xls')
		cursor.close() ; connection.close()
		open(fex1, 'wb').write(blob)
		self.Pro_Bar.SetValue(1)

		xlApp = Dispatch("Excel.Application")   # Connect Excel
		xlApp.Visible=False
		xlWb = xlApp.Workbooks.Open(fex1)
		XLSheet=xlWb.Worksheets("Input_Data")
		sheets = xlWb.Sheets
		self.Pro_Bar.SetValue(2)
		
		PBV=2
		for i, ( SHT, DATA) in enumerate(zip(Lst_Sheets, Lst_data)):
			PBV+=1
			XLSheet=xlWb.Worksheets(SHT)
			sheets(SHT).Activate()
			Put_Clipboard(DATA)
			XLSheet.Cells(3,1).Select()
			XLSheet.Paste()
			XLSheet.Cells(3,1).Select()
			self.Pro_Bar.SetValue(PBV)

		XLSheet.Cells(1,3).Value = self.POSE_Freq
		XLSheet.Cells(1,4).Value = self.POSE_Amp0
		XLSheet.Cells(1,5).Value = self.POSE_Amp
		
		self.Pro_Bar.SetValue(7)
		Put_Clipboard(" ")
		xlWb.Close(SaveChanges=1)
		xlApp.Quit()

		self.Pro_Bar.Hide()
		
		sir="\nThe numerical results were exported to Excel file:\n\n  "+fex1+"\n"
		wx.MessageBox(sir, "Info", wx.OK )
	
	def OnHelp(self, event): 
		fex1=os.getcwd()+"/RESULTS/Readme.txt"
		connection = sqlite.connect(os.getcwd()+'\Config.db') ; cursor = connection.cursor()
		blob=ExtragTextMemory(cursor, 'Readme.txt')
		cursor.close() ; connection.close()
		open(fex1, 'wb').write(blob)						
		
		# Read Help Message from "Readme.txt" file
		with open(fex1, 'r') as f:
			HelpMsj= f.read()
#			for line in f.readlines():
#				print  line.strip()
#				HelpMsj = line.strip()
		os.remove(fex1)
		dlg = wx.lib.dialogs.ScrolledMessageDialog(self, HelpMsj, "Help")
		dlg.ShowModal()
		
		
	
	def OnWord(self, event): # Export results to Microsoft Word file
			
		# 1) Save charts as images
		self.OnSaveImg()

		# 2) Creating a word application object
		wordApp = win32com.client.gencache.EnsureDispatch('Word.Application')
		wordApp.Visible = True 
		doc = wordApp.Documents.Add() # create a new application
		
		# 3) Formating the document
		doc.PageSetup.RightMargin = 20
		doc.PageSetup.LeftMargin = 20
		doc.PageSetup.Orientation = win32com.client.constants.wdOrientLandscape
		doc.PageSetup.PageWidth = 595   # a4 paper size: 595x842
		doc.PageSetup.PageHeight = 842
		header_range= doc.Sections(1).Headers(win32com.client.constants.wdHeaderFooterPrimary).Range
		header_range.ParagraphFormat.Alignment = win32com.client.constants.wdAlignParagraphCenter
		header_range.Font.Bold = True
		header_range.Font.Size = 12
		header_range.Text = time.strftime("%d.%m.%Y")+"   PyFEST Report for   '"+self.OpenFile+"' file"

		# 4) Inserting Tables for images
		total_column = 1
		total_row = len(self.NumeFigura)*2
		rng = doc.Range(0,0)
		rng.ParagraphFormat.Alignment = win32com.client.constants.wdAlignParagraphCenter
		table = doc.Tables.Add(rng,total_row, total_column)
		table.Borders.Enable = False
		if total_column > 1:  table.Columns.DistributeWidth()
		lin=0 ; col=1 ; contor=0
		for File_Img in self.NumeFigura:
				lin=lin+1
				# Formatting the style of each cell
				cell_range= table.Cell(lin, col).Range
				cell_range.ParagraphFormat.LineSpacingRule = win32com.client.constants.wdLineSpaceSingle
				cell_range.ParagraphFormat.SpaceBefore = 0
				cell_range.ParagraphFormat.SpaceAfter = 3
				# Insert the image
				current_pic = cell_range.InlineShapes.AddPicture(File_Img)
				# Putting a name underneath each image
				lin=lin+1
				table.Cell(lin, col).Range.InsertAfter(File_Img)
				contor+=1

	def _TOOLBAR(self):  # Create toolbar
		self.toolbar = self.CreateToolBar( ( wx.TB_HORIZONTAL  | wx.NO_BORDER | wx.TB_FLAT | wx.TB_3DBUTTONS | wx.TB_TEXT) )
		self.toolbar.SetToolBitmapSize((24,24))
		connection = sqlite.connect(os.getcwd()+'\Config.db') ; cursor = connection.cursor()
		Lst_label=["Fit Chart", "Zoom", "Pan", "Data table",  "Word", "Excel","Help", "Exit"]
		Lst_icons=["home.png", "zoom.png", "pan.png", "Show.png", "MsWord.jpg","Excel.png","help.jpg", "exit.png"]
		for i, ( label, icon) in enumerate(zip(Lst_label,Lst_icons )):
			ID_tool=wx.NewId()
			img = ExtragImageMemory(cursor, icon)
			sel_bmp=wx.BitmapFromImage(img.Scale(24,24))
			self.toolbar.AddLabelTool(ID_tool, label, sel_bmp)
			if label=="Fit Chart": self.Bind(wx.EVT_TOOL, self.Home, id=ID_tool)	
			if label=="Zoom": self.Bind(wx.EVT_TOOL, self.Zoom, id=ID_tool)
			if label=="Pan": self.Bind(wx.EVT_TOOL, self.Pan, id=ID_tool)
			if label=="Data table": self.Bind(wx.EVT_TOOL, self.OnTabele, id=ID_tool)				
			if label=="Word": self.Bind(wx.EVT_TOOL, self.OnWord, id=ID_tool)
			if label=="Excel": self.Bind(wx.EVT_TOOL, self.OnExcel, id=ID_tool)
			if label=="Help": self.Bind(wx.EVT_TOOL, self.OnHelp, id=ID_tool)
			if label=="Exit": self.Bind(wx.EVT_TOOL, self.OnClose, id=ID_tool)
		self.toolbar.Realize()   # Icon bar generation
		self.toolbar.Show()
		cursor.close() ; connection.close()
		
	def OnClose(self, event):
		self.Close(True)
		self.Destroy()

#------------------------------------------------
if __name__ == '__main__':
	app = wx.App(redirect=False) # Create the application object
	Calcul_DFT()  # Call "Calcul_DFT" class
	app.MainLoop() # Enter the mainloop
#------------------------------------------------

"""