# -*- coding: ms949 -*-


import random
import wx
import Categorizer
import APP_ResultWindow
import sys
import copy
import matplotlib
import os
import threading
import scipy
from scipy.stats import hypergeom
import numpy
from scipy.stats import norm

Categorizer.CATEGORY_NO_ANNOTATION = 'Uncategorized'
Categorizer.WX_PYTHON = True #  True: show a progress bar

WHITE = wx.Colour ( 255,255,255 )

FLAG = 0

# MPI=False -> cpu=1
MPI = False

# If this is a compiled version, force to set cpu=1.
if os.path.exists('CategorizerGUI.exe'):
	MPI = False
else:
	MPI = True

TEST_BUTTONS = False




DEFAULT_CUTOFF = '0.3'


def create(parent):
	return MainWindow(parent)

[wxID_MAINWINDOW, wxID_MAINWINDOWBUTTON1, wxID_MAINWINDOWBUTTON2,
 wxID_MAINWINDOWBUTTONOPENANNOTATION,
 wxID_MAINWINDOWBUTTONOPENDEF, wxID_MAINWINDOWBUTTONRUN, 
 wxID_MAINWINDOWGAUGEPROGRESS, wxID_MAINWINDOWLABELPERCENT, 
 wxID_MAINWINDOWLISTBOXCATEGORIES, wxID_MAINWINDOWRADIOBUTTONMULTIPLE, 
 wxID_MAINWINDOWRADIOBUTTONSINGLE, wxID_MAINWINDOWSTATICLINE1, 
 wxID_MAINWINDOWSTATICLINE2, wxID_MAINWINDOWSTATICLINE3, 
 wxID_MAINWINDOWSTATICLINE4, wxID_MAINWINDOWSTATICLINE5, 
 wxID_MAINWINDOWSTATICTEXT1, wxID_MAINWINDOWSTATICTEXT2, 
 wxID_MAINWINDOWSTATICTEXT3, wxID_MAINWINDOWSTATICTEXT4, 
 wxID_MAINWINDOWSTATICTEXT5, wxID_MAINWINDOWSTATICTEXT6, 
 wxID_MAINWINDOWSTATICTEXT7, wxID_MAINWINDOWSTATICTEXTCUTOFF, 
 wxID_MAINWINDOWSTATICTEXTURL, wxID_MAINWINDOWSTATICTEXTURL2,
 wxID_MAINWINDOWTEXTBOXSHOWPROGRESS,
 wxID_MAINWINDOWTEXTCTRLANNOTATIONFILE, wxID_MAINWINDOWTEXTCTRLCUTOFF, 
 wxID_MAINWINDOWTEXTCTRLREFGENES, wxID_MAINWINDOWTEXTCTRLUSERGENES,
 wxID_MAINWINDOWBUTTONOPENUSERGENES, wxID_MAINWINDOWBUTTONOPENREFGENES,
 wxID_EVT_RESULT, wxID_CAPTION_CPU, wxID_TEXT_CPU
] = [wx.NewId() for _init_ctrls in range(36)]








################################################
# Thread
def EVT_RESULT(win, func):
	win.Connect(-1, -1, wxID_EVT_RESULT, func)
class ResultEvent(wx.PyEvent):
	"""Simple event to carry arbitrary result data."""
	def __init__(self, data):
		"""Init Result Event."""
		wx.PyEvent.__init__(self)
		self.SetEventType(wxID_EVT_RESULT)
		self.data = data

class ProcessThread(threading.Thread):

	wxObject = None
	targetMethod = None

	def __init__(self, wxObject, targetMethod):
		threading.Thread.__init__(self)

		self.wxObject = wxObject
		self.targetMethod = targetMethod
		self.start()


	def run(self):
		self.targetMethod() # run __process()

		wx.PostEvent(self.wxObject, ResultEvent("Thread finished!"))







class MainWindow(wx.Frame):

	'''
	default values
	'''

	__def_filename = None
	__def_filename_flag = None

	__annot_filename = None
	__annot_filename_flag = None

	__user_genes = None
	__ref_genes = None
	__method = None
	__cutoff = None


	__cache = None
	__thread = None

	__prev_method = -1
	__prev_cutoff = -1
	__cpu = 3



	def _init_ctrls(self, prnt):


		rgb = matplotlib.colors.ColorConverter()
		alpha = 0.3
		YELLOW = rgb.to_rgba( (0.75, 0.75, 0), alpha=alpha)

		wx.Frame.__init__(self, id=wxID_MAINWINDOW, name='MainWindow',
			  parent=prnt, pos=wx.Point(-1, -1), size=wx.Size(950, 422),
			  style=wx.DEFAULT_FRAME_STYLE ^ wx.RESIZE_BORDER ^ wx.MAXIMIZE_BOX,
			  title='Categorizer v1.01')
		self.SetClientSize(wx.Size(934, 403))
		self.SetToolTipString('')
		self.SetBackgroundColour(WHITE)
		self.Bind(wx.EVT_CLOSE, self.OnCloseWindow)

		self.staticText1 = wx.StaticText(id=wxID_MAINWINDOWSTATICTEXT1,
			  label='Step1: Load a category file', name='staticText1',
			  parent=self, pos=wx.Point(16, 16), size=wx.Size(132, 41),
			  style=0)
		self.staticText1.Wrap(132)
		#self.staticText1.SetBackgroundColour(WHITE)

		self.buttonOpenDef = wx.Button(id=wxID_MAINWINDOWBUTTONOPENDEF,
			  label='...', name='buttonOpenDef', parent=self, pos=wx.Point(56,
			  68), size=wx.Size(48, 23), style=0)
		self.buttonOpenDef.Bind(wx.EVT_BUTTON, self.OnButtonOpenDefButton,
			  id=wxID_MAINWINDOWBUTTONOPENDEF)
		#self.buttonOpenDef.SetBackgroundColour( 'yellow' )



		self.listBoxCategories = wx.ListBox(choices=[],
			  id=wxID_MAINWINDOWLISTBOXCATEGORIES, name='listBoxCategories',
			  parent=self, pos=wx.Point(16, 105), size=wx.Size(132, 264),
			  style=wx.LC_REPORT|wx.BORDER_SUNKEN)
		self.listBoxCategories.SetBackgroundColour('yellow')

		self.staticLine1 = wx.StaticLine(id=wxID_MAINWINDOWSTATICLINE1,
			  name='staticLine1', parent=self, pos=wx.Point(153, 10),
			  size=wx.Size(7, 359), style=0)

		self.staticText2 = wx.StaticText(id=wxID_MAINWINDOWSTATICTEXT2,
			  label='Step2: Load an annotation file', name='staticText2',
			  parent=self, pos=wx.Point(174, 16), size=wx.Size(88, 32),
			  style=0)
		self.staticText2.Wrap(88)
		#self.staticText2.SetBackgroundColour(WHITE)

		self.textCtrlAnnotationFile = wx.TextCtrl(id=wxID_MAINWINDOWTEXTCTRLANNOTATIONFILE,
			  name='textCtrlAnnotationFile', parent=self, pos=wx.Point(168, 105),
			  size=wx.Size(108, 264), style=wx.TE_MULTILINE | wx.TE_READONLY,
			  value='<None>')
		self.textCtrlAnnotationFile.SetBackgroundColour('yellow')

		self.buttonOpenAnnotation = wx.Button(id=wxID_MAINWINDOWBUTTONOPENANNOTATION,
			  label='...', name='buttonOpenAnnotation', parent=self,
			  pos=wx.Point(200, 68), size=wx.Size(48, 23), style=0)
		self.buttonOpenAnnotation.Bind(wx.EVT_BUTTON,
			  self.OnButtonOpenAnnotationButton,
			  id=wxID_MAINWINDOWBUTTONOPENANNOTATION)
		#self.buttonOpenAnnotation.SetBackgroundColour('yellow')


		self.staticLine2 = wx.StaticLine(id=wxID_MAINWINDOWSTATICLINE2,
			  name='staticLine2', parent=self, pos=wx.Point(280, 8),
			  size=wx.Size(8, 361), style=0)

		self.staticText3 = wx.StaticText(id=wxID_MAINWINDOWSTATICTEXT3,
			  label='Step3: Enter your genes', name='staticText3', parent=self,
			  pos=wx.Point(296, 16), size=wx.Size(119, 40), style=0)
		self.staticText3.Wrap(119)

		self.staticText4 = wx.StaticText(id=wxID_MAINWINDOWSTATICTEXT4,
			  label='Step4: Enter background genes (optional)', name='staticText4',
			  parent=self, pos=wx.Point(450, 16), size=wx.Size(100, 63),
			  style=0)
		self.staticText4.Wrap(100)

		# Text box (user's genes)
		self.textCtrlUserGenes = wx.TextCtrl(id=wxID_MAINWINDOWTEXTCTRLUSERGENES,
			  name='textCtrlUserGenes', parent=self, pos=wx.Point(296, 105),
			  size=wx.Size(120, 264), style=wx.TE_MULTILINE , value='<None>') # | wx.TE_READONLY
		self.textCtrlUserGenes.SetBackgroundColour('yellow')

		# Button
		self.buttonOpenUserGenes = wx.Button(id=wxID_MAINWINDOWBUTTONOPENUSERGENES,
			  label='...', name='buttonOpenUserGenes', parent=self,
			  pos=wx.Point(331, 68), size=wx.Size(48, 23), style=0)
		self.buttonOpenUserGenes.Bind(wx.EVT_BUTTON,
			  self.OnButtonOpenUserGenesButton,
			  id=wxID_MAINWINDOWBUTTONOPENUSERGENES)
		#self.buttonOpenUserGenes.SetBackgroundColour('yellow')


		# Text box for background genes
		self.textCtrlRefGenes = wx.TextCtrl(id=wxID_MAINWINDOWTEXTCTRLREFGENES,
		    name='textCtrlRefGenes', parent=self, pos=wx.Point(440, 105),
		    size=wx.Size(128, 264), style=wx.TE_MULTILINE, value='<None>') #| wx.TE_READONLY

		# Button
		self.buttonOpenRefGenes = wx.Button(id=wxID_MAINWINDOWBUTTONOPENREFGENES,
			  label='...', name='buttonOpenRefGenes', parent=self,
			  pos=wx.Point(475, 68), size=wx.Size(48, 23), style=0)
		self.buttonOpenRefGenes.Bind(wx.EVT_BUTTON,
			  self.OnButtonOpenRefGenesButton,
			  id=wxID_MAINWINDOWBUTTONOPENREFGENES)
		#self.buttonOpenUserGenes.SetBackgroundColour('yellow')



		self.staticLine3 = wx.StaticLine(id=wxID_MAINWINDOWSTATICLINE3,
			  name='staticLine3', parent=self, pos=wx.Point(424, 10),
			  size=wx.Size(8, 361), style=0)

		self.staticLine4 = wx.StaticLine(id=wxID_MAINWINDOWSTATICLINE4,
			  name='staticLine4', parent=self, pos=wx.Point(576, 8),
			  size=wx.Size(8, 361), style=0)

		self.staticText5 = wx.StaticText(id=wxID_MAINWINDOWSTATICTEXT5,
			  label='Step5: Options', name='staticText5', parent=self,
			  pos=wx.Point(600, 16), size=wx.Size(100, 13), style=0)
		#self.staticText5.Wrap(100)

		self.radioButtonSingle = wx.RadioButton(id=wxID_MAINWINDOWRADIOBUTTONSINGLE,
			  label='Single category', name='radioButtonSingle', parent=self,
			  pos=wx.Point(590, 104), size=wx.Size(134, 18), style=0)
		self.radioButtonSingle.SetValue(False) 
		self.radioButtonSingle.SetToolTipString(u'radioButtonSingle')
		self.radioButtonSingle.Bind(wx.EVT_RADIOBUTTON,
			  self.OnRadioButtonSingleRadiobutton,
			  id=wxID_MAINWINDOWRADIOBUTTONSINGLE)

		self.radioButtonMultiple = wx.RadioButton(id=wxID_MAINWINDOWRADIOBUTTONMULTIPLE,
			  label='Multiple categories', name='radioButtonMultiple',
			  parent=self, pos=wx.Point(590, 128), size=wx.Size(150, 18),
			  style=0)
		self.radioButtonMultiple.SetValue(True)
		self.radioButtonMultiple.Bind(wx.EVT_RADIOBUTTON,
			  self.OnRadioButtonMultipleRadiobutton,
			  id=wxID_MAINWINDOWRADIOBUTTONMULTIPLE)

		self.staticText6 = wx.StaticText(id=wxID_MAINWINDOWSTATICTEXT6,
			  label='A gene can be included in', name='staticText6',
			  parent=self, pos=wx.Point(592, 60), size=wx.Size(123, 15),
			  style=0)
		self.staticText6.Wrap(123)

		self.staticTextCutoff = wx.StaticText(id=wxID_MAINWINDOWSTATICTEXTCUTOFF,
			  label='Cutoff', name='staticTextCutoff', parent=self,
			  pos=wx.Point(590, 160), size=wx.Size(38, 13), style=0)


		global DEFAULT_CUTOFF

		self.textCtrlCutoff = wx.TextCtrl(id=wxID_MAINWINDOWTEXTCTRLCUTOFF,
			  name='textCtrlCutoff', parent=self, pos=wx.Point(640, 160),
			  size=wx.Size(40, 24), style=0, value=DEFAULT_CUTOFF)
		#self.textCtrlCutoff.Enable(False)


		# cpu option
		global MPI

		self.staticTextCPU = wx.StaticText(id=wxID_CAPTION_CPU,
		                                   name = 'caption_cpu',
		                                   parent=self,
		                                   pos = wx.Point(590, 200),
		                                   size = wx.Size(30,13),
		                                   style = 0,
		                                   label = 'CPU'
		                                   )
		self.textCtrlCPU = wx.TextCtrl(id = wxID_TEXT_CPU,
		                               name = 'text_cpu',
		                               parent = self,
		                               pos = wx.Point(630, 200),
		                               size = wx.Size(40,24),
		                               style = 0,
		                               value = str(self.__cpu).strip()
		                               )

		if not MPI:
			self.staticTextCPU.Hide()
			self.textCtrlCPU.Hide()
			self.textCtrlCPU.Value = "1"

		# -----------


		self.staticLine5 = wx.StaticLine(id=wxID_MAINWINDOWSTATICLINE5,
			  name='staticLine5', parent=self, pos=wx.Point(741, 10),
			  size=wx.Size(7, 334), style=0)




		# ------------


		self.buttonRun = wx.Button(id=wxID_MAINWINDOWBUTTONRUN, label='Run',
			  name='buttonRun', parent=self, pos=wx.Point(764, 80),
			  size=wx.Size(75, 23), style=0)
		self.buttonRun.Bind(wx.EVT_BUTTON, self.OnButtonRunButton,
			  id=wxID_MAINWINDOWBUTTONRUN)
		self.buttonRun.SetBackgroundColour('yellow')





		self.staticText7 = wx.StaticText(id=wxID_MAINWINDOWSTATICTEXT7,
			  label='Step6: Run!', name='staticText7', parent=self,
			  pos=wx.Point(764, 16), size=wx.Size(100, 30), style=0)



		self.gaugeProgress = wx.Gauge(id=wxID_MAINWINDOWGAUGEPROGRESS,
			  name='gaugeProgress', parent=self, pos=wx.Point(764, 160),
			  range=100, size=wx.Size(144, 16), style=wx.GA_HORIZONTAL)
		self.gaugeProgress.SetToolTipString('Progress')

		self.textBoxShowProgress = wx.StaticText(id=wxID_MAINWINDOWTEXTBOXSHOWPROGRESS,
			  label='Idle...', name='textBoxShowProgress', parent=self,
			  pos=wx.Point(764, 134), size=wx.Size(300, 15), style=0)

		self.labelPercent = wx.StaticText(id=wxID_MAINWINDOWLABELPERCENT,
			  label='0 %', name='labelPercent', parent=self, pos=wx.Point(820,
			  176), size=wx.Size(80, 13), style=0)
		self.labelPercent.SetBackgroundStyle(wx.BG_STYLE_SYSTEM)
		self.labelPercent.SetToolTipString('')



		self.staticTextURL = wx.StaticText(id=wxID_MAINWINDOWSTATICTEXTURL,
			  label='http://ssbio.cau.ac.kr/software/categorizer', name='staticTextURL', parent=self,
			  pos=wx.Point(650, 345), size=wx.Size(270, 15), style=0)
		self.staticTextURL.SetBackgroundStyle(wx.BG_STYLE_SYSTEM)
		self.staticTextURL.SetForegroundColour(wx.Colour(0, 0, 255))
		self.staticTextURL.Bind(wx.EVT_LEFT_DOWN, self.OnStaticTextURLLeftDown)


		self.staticTextURL2 = wx.StaticText(id=wxID_MAINWINDOWSTATICTEXTURL2,
			  label='http://chibi.ubc.ca/categorizer', name='staticTextURL2', parent=self,
			  pos=wx.Point(650, 360), size=wx.Size(200, 15), style=0)
		self.staticTextURL2.SetBackgroundStyle(wx.BG_STYLE_SYSTEM)
		self.staticTextURL2.SetForegroundColour(wx.Colour(0, 0, 255))
		self.staticTextURL2.Bind(wx.EVT_LEFT_DOWN, self.OnStaticTextURLLeftDown2)


		self.menu_bar = wx.MenuBar()

		self.menu_menu = wx.Menu()
		about = self.menu_menu.Append( wx.ID_ABOUT, 'About')
		quit = self.menu_menu.Append( wx.ID_EXIT, 'Quit')

		self.menu_bar.Append( self.menu_menu, 'Menu')
		self.SetMenuBar(self.menu_bar)

		self.Bind( wx.EVT_MENU, self.OnUpdateMenuAbout, about)
		self.Bind( wx.EVT_MENU, self.OnUpdateMenuQuit, quit)


		# event
		EVT_RESULT(self, self.updateDisplay)


		# debug
		global TEST_BUTTONS
		if TEST_BUTTONS:
			self.button1 = wx.Button(id=wxID_MAINWINDOWBUTTON1, label='small',
				  name='button1', parent=self, pos=wx.Point(824, 272),
				  size=wx.Size(75, 23), style=0)
			self.button1.Bind(wx.EVT_BUTTON, self.OnButton1Button,
				  id=wxID_MAINWINDOWBUTTON1)

			self.button1 = wx.Button(id=wxID_MAINWINDOWBUTTON2, label='big',
				  name='button2', parent=self, pos=wx.Point(824, 312),
				  size=wx.Size(75, 23), style=0)
			self.button1.Bind(wx.EVT_BUTTON, self.OnButton2Button,
				  id=wxID_MAINWINDOWBUTTON2)



		# icon
		path = os.path.abspath("./cat.ico")
		icon = wx.Icon(path, wx.BITMAP_TYPE_ICO)
		self.SetIcon(icon)

	def ask(self, msg):

		result = wx.MessageBox(msg, style = wx.CENTER | wx.ICON_QUESTION | wx.YES_NO )
		if result == wx.YES:
			return True
		else:
			return False


	def OnStaticTextURLLeftDown(self, event):
		# open a browser
		import webbrowser
		webbrowser.open( self.staticTextURL.Label)


	def OnStaticTextURLLeftDown2(self, event):
		#  open a browser
		import webbrowser
		webbrowser.open( self.staticTextURL2.Label)


	def OnUpdateMenuAbout(self, event):

		evt_id = event.GetId()

		if evt_id == wx.ID_ABOUT:
			self.about()

		else:
			event.Skip()


	def getHypergeomPvalue(self, success_in_sample, sample_number, success_in_pop, pop_number):

		p = hypergeom.pmf( numpy.arange(success_in_sample, sample_number+1), pop_number, success_in_pop , sample_number).sum()
		return p

	def getPvaluefromZscore(self, value, mean, std):
		z = abs( float(( mean - value )/float(std) ) )
		p_value = norm.sf(z) * 2
		return p_value
	
	def get_pvalue_of_fisher_exact_test(self, table, option = 'two-sided'):
		'''
		option=two-sided, less, greater
		table은 2x2 여야하고
	
	
			A   B
		C   1   5
		D   4   8
	
		위의 table이라면 table = [ [1,5], [4,8] ] 이렇게 넣어야 한다.
		'''
	
		oddsration, pvalue = scipy.stats.fisher_exact(table, option)
		return pvalue		
		
	def about(self):

		info = wx.AboutDialogInfo()
		desc = 'Categorizer v1.01\nBug report: dna@ssbio.cau.ac.kr'
		py_version = [sys.platform, ", python ", sys.version.split()[0] ]
		platform = list(wx.PlatformInfo[1:])
		platform[0] += (" " + wx.VERSION_STRING)
		wx_info = ', '.join(platform)


		info.SetName('Categorizer')
		info.SetVersion('1.01')



		desc = [
			'Your platform: ' + sys.platform,
		    'http://chibi.ubc.ca/categorizer',
			'http://ssbio.cau.ac.kr/software/categorizer',
			'Bug report: dna@ssbio.cau.ac.kr'
			]


		info.SetDescription( '\n'.join(desc) )

		wx.AboutBox(info)

	def OnUpdateMenuQuit(self, event):

		evt_id = event.GetId()

		if evt_id == wx.ID_EXIT:
			#  quit
			self.quitMe()


	def __init__(self, parent):



		self._init_ctrls(parent)

		self.__cache = {}
		self.__def_filename = None
		self.__def_filename_flag = False # error?
		self.__annot_filename = None
		self.__annot_filename_flag = False # error?
		self.__user_genes = []
		self.__ref_genes = []
		self.__method = 0 # single category=0, multiple categories=1
		self.__cutoff = 0.2
		self.__result = None
		self.__thread = None
		self.__prev_method = -1
		self.__prev_cutoff = -1


		global MPI
		if MPI == False:
			self.__cpu = 1


	def OnListBox1Listbox(self, event):
		event.Skip()



	def OnRadioButtonSingleRadiobutton(self, event):
		#self.__cache = {}
		#self.__toggleRadioButton()
		event.Skip()

	def OnRadioButtonMultipleRadiobutton(self, event):
		#self.__cache = {}
		#self.__toggleRadioButton()
		event.Skip()

	def OnButtonRunButton(self, event):
		self.__run()

	def OnButtonOpenDefButton(self, event):

		# empty cache
		self.__cache = {}


		# open a category file
		wildcard = 'All files|*.*'
		dlg = wx.FileDialog(self,
							message="Open a category file",
							wildcard = wildcard,
							style = wx.FD_OPEN)

		if dlg.ShowModal() == wx.ID_OK:
			path = dlg.GetPath()

			self.__def_filename = path
			self.__def_filename_flag = self.__checkDefFile( path )


		dlg.Destroy()

		event.Skip()

	def OnButtonOpenUserGenesButton(self, event):
		lst = self.__getGenes("Open a gene list file")
		if len(lst) != 0:
			self.textCtrlUserGenes.Value = '\n'.join(lst)


	def OnButtonOpenRefGenesButton(self, event):
		lst = self.__getGenes("Open a background gene list file")
		if len(lst) != 0:
			self.textCtrlRefGenes.Value = '\n'.join(lst)

	def __getGenes(self, msg):

		lst = []

		wildcard = 'All files|*.*'
		dlg = wx.FileDialog(self,
							message=msg,
							wildcard = wildcard,
							style = wx.FD_OPEN)

		if dlg.ShowModal() == wx.ID_OK:
			fname = dlg.GetPath()
			lst = self.__loadGenes(fname)

		return lst

	def __loadGenes(self, fname):

		lst = {}
		try:
			f=open(fname,'r')

			for s in f.readlines():
				s = s.replace('\n','').strip()
				if len(s) == 0: continue

				lst[s] = None

		except:
			lst = {}
			wx.MessageBox("Found IO error while reading " + fname, caption="Error", style=wx.OK|wx.CENTER|wx.ICON_ERROR, parent=None, x=-1, y=-1)
		finally:
			f.close()


		return lst.keys()


	def OnButtonOpenAnnotationButton(self, event):

		self.__cache = {}


		# load an annotation file
		wildcard = 'All files|*.*'
		dlg = wx.FileDialog(self,
							message="Open an GO annotation file",
							wildcard = wildcard,
							style = wx.FD_OPEN)

		if dlg.ShowModal() == wx.ID_OK:
			path = dlg.GetPath()
			self.__annot_filename = path
			self.__annot_filename_flag = self.__checkAnnotationFile(path)


		dlg.Destroy()

		event.Skip()







	def __toggleRadioButton(self):


		# toggle buttons
		if self.radioButtonSingle.Value:
			self.textCtrlCutoff.Enabled = False
		else:
			self.textCtrlCutoff.Enabled = True


	def quitMe(self):
		if self.ask("Do you want to quit?"):

			if self.__thread is not None:
				Categorizer.THREAD_TERMINATION = True
				self.__thread.join()

			sys.exit(1)



	def OnCloseWindow(self, event):
		evt_id = event.GetId()
		self.quitMe()



	def __buttonsOn(self):
		# enable all buttons
		self.__buttons(True)


	def __buttonsOff(self):
		# disable all buttons
		self.__buttons(False)

	def __buttons(self, state):
		self.buttonOpenAnnotation.Enabled = state
		self.buttonOpenDef.Enabled = state
		self.buttonOpenRefGenes.Enabled = state
		self.buttonOpenUserGenes.Enabled = state

		self.radioButtonMultiple.Enabled = state
		self.radioButtonSingle.Enabled = state
		self.textCtrlCutoff.Enabled = state

		self.buttonRun.Enabled = state

		self.textCtrlCPU.Enabled = state


	def __checkIndexFiles(self):

		# go_index.txt, go_prob.txt
		if os.path.exists('go_index.txt') and os.path.exists('go_prob.txt'):
			return True
		else:
			wx.MessageBox("go_index.txt or go_prob.txt not found. If not, run Preprocessor.py to generate them.", caption="Error", style=wx.OK|wx.CENTER|wx.ICON_ERROR, parent=None, x=-1, y=-1)
			return False

	def __run(self):

		if self.__checkDefFile( self.__def_filename ) and \
		   self.__checkAnnotationFile( self.__annot_filename   ) and \
		   self.__checkUserGenes() and \
		   self.__checkRefGenes() and \
		   self.__checkOptions() and \
		   self.__checkCPUOption() and \
		   self.__checkIndexFiles():




			self.buttonRun.Enabled = False
			self.gaugeProgress.Value = 0
			self.labelPercent.Label = '0 %'

			self.__buttonsOff()


			# when thread is over, run  updateDisplay method
			self.__thread = ProcessThread(self, self.__process)




	def __clearDefList(self):
		self.listBoxCategories.Clear()




	def __checkDefFile(self, fname):

		self.__clearDefList()


		r = {}

		try:
			r = self.__openDefFile(fname)
			if type(r) is str:
				#  Encountered an error while loading afile
				wx.MessageBox("Step1: " + r,
				              caption="Error",
				              style=wx.OK|wx.CENTER|wx.ICON_ERROR,
				              parent=None, x=-1, y=-1)
				return False


		except:
			# IO error
			wx.MessageBox("Step1: Incorrect category file!", caption="Error", style=wx.OK|wx.CENTER|wx.ICON_ERROR, parent=None, x=-1, y=-1)
			return False


		# No categories?
		if len(r) == 0:
			wx.MessageBox("Step1: No categories are found!", caption="Error", style=wx.OK|wx.CENTER|wx.ICON_ERROR, parent=None, x=-1, y=-1)
			return False


		key = r.keys()
		key.sort()

		s = []
		e = []
		for k in key:
			txt = k + '(' + str(len(r[k])) + ')'
			s.append(txt)

			self.listBoxCategories.Append(txt)


			if len( r[k] ) == 0:
				e.append(k)


		if len(e)>0:
			# Error
			txt = 'No GO terms are in ' + ', '.join(e)
			wx.MessageBox("Step1: " + txt, caption="Error", style=wx.OK|wx.CENTER|wx.ICON_ERROR, parent=None, x=-1, y=-1)
			return False



		return True

	def __checkAnnotationFile(self, fname):
		self.textCtrlAnnotationFile.Value = '<None>'


		r = {}

		try:
			r = self.__openAnnotationFile(fname)
		except:
			wx.MessageBox("Step2: Incorrect annotation file!", caption="Error", style=wx.OK|wx.CENTER|wx.ICON_ERROR, parent=None, x=-1, y=-1)
			return False


		# No annotations?
		if len(r) == 0:
			wx.MessageBox("Step2: No annotations are found!", caption="Error", style=wx.OK|wx.CENTER|wx.ICON_ERROR, parent=None, x=-1, y=-1)
			return False


		#self.textCtrlAnnotationFile.Value = 'Loaded successfully from <'+fname+'>'


		txt = 'Loaded ' + str(len(r)) + ' genes\n\n'
		txt += '\n'.join(r.keys())
		self.textCtrlAnnotationFile.Value = txt
		
		
		
		return True




	def __openAnnotationFile(self, filename):


		gene_goid_dict = {}
		init = True

		f=open(filename,'r')

		for s in f.readlines():

			s = s.replace('\n','').replace('\r','')

			if len(s) == 0: continue
			if s[0] == '!': continue
			if init:
				init = False
				continue


			x = s.split('\t')

			db_name = x[0].strip()
			uid = x[1].strip()
			name = x[2].strip()
			go_id = x[4].strip()
			#category = x[8].strip()
			#desc = x[9].strip()

			# unique id
			if not gene_goid_dict.has_key(uid):
				gene_goid_dict[ uid ] = []

			if not go_id in gene_goid_dict[uid]:
				gene_goid_dict[uid].append(go_id)

			# gene id
			if not gene_goid_dict.has_key(name):
				gene_goid_dict[ name ] = []

			if not go_id in gene_goid_dict[name]:
				gene_goid_dict[name].append(go_id)



		f.close()

		return gene_goid_dict


	def __getGenesFromText(self, txt):


		r = {}

		for s in txt.split('\n'):
			s = s.strip()

			if len(s) == 0: continue
			if s == '<None>': continue
			r[s] = None

		return r.keys()

	def __checkUserGenes(self):

		self.__user_genes = self.__getGenesFromText( self.textCtrlUserGenes.Value)

		if len(self.__user_genes) == 0:

			wx.MessageBox("Step3: No user genes entered!", caption="Error", style=wx.OK|wx.CENTER|wx.ICON_ERROR, parent=None, x=-1, y=-1)
			return False


		return True

	def __checkRefGenes(self):

		self.__ref_genes = self.__getGenesFromText( self.textCtrlRefGenes.Value)


		if len( self.__ref_genes ) > 0:
			if len( self.__ref_genes ) < len( self.__user_genes ):
				wx.MessageBox("Step4: Number of background genes must be larger than that of Step 3!", caption="Error", style=wx.OK|wx.CENTER|wx.ICON_ERROR, parent=None, x=-1, y=-1)
				return False


		return True


	def __checkCPUOption(self):

		v = 1

		try:
			v = eval( self.textCtrlCPU.Value )
		except:
			wx.MessageBox("Step5: Incorrect cpu value!", caption="Error", style=wx.OK|wx.CENTER|wx.ICON_ERROR, parent=None, x=-1, y=-1)
			return False

		self.__cpu = v
		return True

	def __checkOptions(self):


		if self.radioButtonMultiple.Value:

			self.__method = 1

		else:
			self.__method = 0


		value = None

		# 0<cutoff<=1
		try:
			value = eval( self.textCtrlCutoff.Value )
		except:
			wx.MessageBox("Step5: Incorrect cutoff value!", caption="Error", style=wx.OK|wx.CENTER|wx.ICON_ERROR, parent=None, x=-1, y=-1)
			return False

		if 0.0 <= value <= 1.0:
			self.__cutoff = value
			return True
		else:
			wx.MessageBox("Step5: Cutoff value must be in the range of 0.0 - 1.0", caption="Error", style=wx.OK|wx.CENTER|wx.ICON_ERROR, parent=None, x=-1, y=-1)
			return False
		return True



		'''
		if self.radioButtonMultiple.Value:

			self.__method = 1
			value = None

			# 0<cutoff<=1
			try:
				value = eval( self.textCtrlCutoff.Value )
			except:
				wx.MessageBox("Step5: Incorrect cutoff value!", caption="Error", style=wx.OK|wx.CENTER|wx.ICON_ERROR, parent=None, x=-1, y=-1)
				return False

			if 0.0 <= value <= 1.0:
				self.__cutoff = value
				return True
			else:
				wx.MessageBox("Step5: Cutoff value must be in the range of 0.0 - 1.0", caption="Error", style=wx.OK|wx.CENTER|wx.ICON_ERROR, parent=None, x=-1, y=-1)
				return False
		else:
			self.__method = 0

		return True
		'''






	def __openDefFile(self, fname):

		r = {}
		f=open(fname,'r')

		current_category = None
		cnt = 0

		for s in f.readlines():
			s = s.replace('\n','').strip().replace('\t','')
			cnt += 1


			if len(s)>0:
				comment_index = s.find('#')

				line = ''

				if comment_index < 0:
					line = s
				elif comment_index == 0:
					continue
				else:
					line = s[ :comment_index].strip()

				if len(line)>0:
					if line[0] == '-':
						# category
						current_category = line[1:].strip()
						r[current_category] = []
					else:
						# GO ID
						if line.find('GO:') == 0:
							if current_category is None:
								msg = 'Line number: ' + str(cnt) + '\n' + \
									'Unidentified GO ID: ' + s

								return msg


							else:
								if not line in r[current_category]:
									r[current_category].append(line)

						else:

							msg = 'Line number: ' + str(cnt) + '\n' + \
								'Unidentified GO ID: ' + s

							return msg




		f.close()



		key = r.keys()
		key.sort()

		#print 'Category # = ', len(key)
		#for k in key:
		#	print k, ' -> ', len(r[k]), 'terms'


		return r






	def __process(self):


		all_categories, gene_category, ref_category = self.__categorize()

		if all_categories is None:
			# Probably user cancelled.
			return


		pie_chart , p_enrichment = self.__analyze(all_categories, gene_category, ref_category)
		self.__result = [ all_categories, gene_category, pie_chart, p_enrichment]

		# when done, run 'updateDisplay'
		self.textBoxShowProgress.Label = 'Done..'
		self.textBoxShowProgress.Refresh()



	def __analyze(self, all_categories, gene_category, ref_category):


		self.textBoxShowProgress.Label = 'Analyzing..'
		self.textBoxShowProgress.Refresh()

		pie_chart = self.__pieChart(all_categories, gene_category)
		p_enrichment = None
		if len(ref_category)>0:
			p_enrichment = self.__enrichment(all_categories, gene_category, ref_category)

		return pie_chart, p_enrichment


	def __pieChart(self, all_categories, gene_category):

		# pie chart

		r = {}
		for k in all_categories:
			r [ k ] = 0.0

		#print 'All categories = ', repr(all_categories)


		for g in gene_category.keys():
			#print g, gene_category[g]

			for cat in gene_category[g].keys():
				r[ cat ] += 1.0

		return r


	def __enrichment(self, all_categories, gene_category, ref_category):

		

		# gene_category
		#     key = gene
		#     value = { category: score }


		user = self.__pieChart( all_categories, gene_category)
		ref = self.__pieChart( all_categories, ref_category)


		# p-values
		user_sum = self.__sum( user, ignore_uncategorized = False )
		ref_sum = self.__sum( ref, ignore_uncategorized = False )

		p = {}
		
		print '-------------------'
		
		# ref를 랜덤하게 만들어서 평균 개수를 구한다.
		# rnd_model: key=category, value=[mean, stdev]
		
		print 'Testing enrichment...'
		
		iteration = 100
		
		rnd_model = self.__randomizeReference(
	                iteration, 
	                all_categories,
	                gene_category,
		        ref_category,
		        user,
		        ref)
		
		
		# --------------------------------
		
		
		
	        
		
		for c in all_categories:
			
			if c == Categorizer.CATEGORY_NO_ANNOTATION:
				continue
			
			value = user[c]
			mean, std = rnd_model[c]
			
			#print c, 'user=', value, 'ref mean=',mean, 'ref std=',std
			
			p[c] = self.getPvaluefromZscore(value, mean, std)
			if value < mean:
				p[c] = 1.0
				
				

		return p


	def makeReferenceProbability(self, cnt):
		
		total = 0.0
		p = {}
		for c in cnt.keys():
			total += cnt[c]
			p[c] = cnt[c]
			
		for c in p.keys():
			p[c] = float(p[c])/float(total)
			
		return p


	def pickCategory(self, background_probability, n):
		
		
		box = []
		
		while(len(box)<n):
			t = 0.0
			r = random.random()
			
			for c in background_probability.keys():
				t += background_probability[c]
				if r <= t:
					if not c in box:
						box.append(c)
					break
		return box

	def __pick_a_gene_and_its_category_randomly(self, ref_category):
		
		gene = random.sample(ref_category.keys(), 1)[0]
		category = random.sample(ref_category[gene].keys(), 1) [0]
		
		return gene, category


	def __mix_up_annotations(self, all_categories, ref_category):
		# randomize gene-GO annotations
		# ref_category
		#     key = gene
		#     value = { category: score }
		
		new_ref = copy.deepcopy(ref_category)
		genes = new_ref.keys()
		
		max_cnt = 2
		
		for g1 in range(len(genes)):
			
			# gene_index - current index
			gene1 = genes[g1]
			
			categories1 = new_ref[gene1].keys()
			
			cnt = 0
			
			for category_index1 in range(len(categories1)):
				category1 = categories1[category_index1]
				
				# do not randomize 'uncategorized'
				if category1 == Categorizer.CATEGORY_NO_ANNOTATION: 
					continue
				
				# pick a gene randomly
				# pick a 
				r_gene, r_category = None, None
				fail = False
				while(True):
					r_gene, r_category = self.__pick_a_gene_and_its_category_randomly(new_ref)
					#if r_gene != gene1 and category1 != r_category:
					if r_gene != gene1 and \
					   r_category != Categorizer.CATEGORY_NO_ANNOTATION and \
					   new_ref[r_gene].has_key(category1) == False and \
					   new_ref[gene1].has_key(r_category) == False:
					   
						break
					
					cnt += 1
					
					if cnt >= max_cnt:
						fail = True
						break
					
				# swap
				if fail == False:
					score1 = new_ref[gene1][category1]
					r_score = new_ref[r_gene][r_category]
					
					#
					#score1 = new_ref[gene1].pop(category1, None)
					#r_score = new_ref[r_gene].pop(r_category, None)
					del new_ref[gene1][category1]
					del new_ref[r_gene][r_category]
					
					new_ref[gene1][r_category] = r_score
					new_ref[r_gene][category1] = score1
				
			
		return new_ref
			
		
		
		
	def __randomizeReference(self,
		                iteration, 
		                all_categories,
		                gene_category,
		                ref_category,
		                user,
		                ref):
			
			
		rnd_model = {}
		total_count = {}
		
		
		
		
		
		
		for c in all_categories:
			rnd_model [c] = None
			total_count [c] = []
		
		# background probability
		#background_probability = self.makeReferenceProbability(ref)
			
		prev_per = -1
		for i in range(iteration):
			
			print '\rRandomization=', i+1, '/', iteration,
			
			per = int( float(i+1)/float(iteration)  * 100 )


			#self.textBoxShowProgress.Label = 'Analyzing..'
			if prev_per != per:
				self.gaugeProgress.Value = per
				self.gaugeProgress.Refresh()
				
				prev_per = per
				
				self.labelPercent.Label = str(per)+' %'
				self.labelPercent.Refresh()
			
			
			count = {}
			for c in all_categories:
				count[c] = 0.0
			
			
			new_ref = self.__mix_up_annotations(all_categories, ref_category)
			# calculate the number of randomized categories of gene_category
			
			for g in gene_category.keys():
				if new_ref.has_key(g):
					for c in new_ref[g]:
						count[c] += 1.0

		
			for c in all_categories:
				total_count[c].append(count[c])
				
		print 'done'
		
		for c in all_categories:
			n = numpy.array(total_count[c])
			mean = n.mean()
			stdev = n.std()
			rnd_model[c] = [mean, stdev]
			
		
		return rnd_model
	
	
	
	'''
	def __randomizeReference(self,
	                iteration, 
	                all_categories,
	                gene_category,
		        ref_category,
		        user,
		        ref):
		
		
		rnd_model = {}
		total_count = {}
		
		
		
		
		
		
		for c in all_categories:
			rnd_model [c] = None
			total_count [c] = []
		
		# background probability
		background_probability = self.makeReferenceProbability(ref)
			
			
		for i in range(iteration):
			
			count = {}
			for c in all_categories:
				count[c] = 0.0
			
			for g in gene_category.keys():
				no_of_categories = len(gene_category[g].keys())
				random_categories = self.pickCategory(background_probability, no_of_categories)
				
				for c2 in random_categories:
					count[c2] += 1.0
		
			for c in all_categories:
				total_count[c].append(count[c])
				
		
		for c in all_categories:
			n = numpy.array(total_count[c])
			mean = n.mean()
			stdev = n.std()
			rnd_model[c] = [mean, stdev]
			
		
		return rnd_model
		
	'''		
			
			

	def __sum(self, dic, ignore_uncategorized = False):
		total = 0.0
		for k in dic.keys():
			
			
			if k != Categorizer.CATEGORY_NO_ANNOTATION or ignore_uncategorized == False:
				total += dic[k]
				
		return total


	def updateDisplay(self, msg):

		# show up a result window

		self.__thread = None

		all_categories, gene_category, pie_chart, p_enrichment = self.__result
		r_window = APP_ResultWindow.create(self)

		r_window.process(all_categories, gene_category, pie_chart, p_enrichment)

		r_window.Show()
		r_window.SetFocus()
		r_window.MakeModal(True)

		self.__buttonsOn()
		self.SetFocus()


	def __categorize(self):


		global MPI





		self.textBoxShowProgress.Label = 'Preprocessing'
		self.textBoxShowProgress.Refresh()

		def_file = self.__def_filename
		user_genes = self.__user_genes
		ref_genes = self.__ref_genes
		annot_file = self.__annot_filename
		method = self.__method
		threshold = self.__cutoff
		cpu = self.__cpu
		
		# cpu개수가 gene개수보다 많을 수는 없다.
		
		if cpu > len(user_genes):
			cpu = len(user_genes)
			
		#if m > len(ref_genes) and len(ref_genes) != 0:
		#	m = len(ref_genes)
		#if m != cpu:
		#	cpu = m

		org_goid_dict = Categorizer.loadAnnotationFile(annot_file)
		cat_def = Categorizer.loadGOcategoryDefinitionFile(def_file)
		ccc = cat_def.keys()
		ccc.sort()
		all_categories = ccc + [ Categorizer.CATEGORY_NO_ANNOTATION ]






		option = {
				Categorizer.OPT_METHOD: None,
				Categorizer.OPT_METHOD_THRESHOLD: threshold,
				Categorizer.OPT_PROGRESS_BAR: self.gaugeProgress,
				Categorizer.OPT_BASAL_COUNT: 0,
				Categorizer.OPT_MAX_COUNT: len(user_genes) + len(ref_genes),
				Categorizer.OPT_PROGRESS_BAR_TEXT: self.labelPercent,
		                Categorizer.OPT_CPU: cpu
				}


		#print '---------------------'
		#print ' OPTION '
		#print repr(option)
		#print '---------------------'

		if method == 0: # single
			option[Categorizer.OPT_METHOD] = Categorizer.OPT_METHOD_SINGLE
		elif method == 1: # multiple
			option[Categorizer.OPT_METHOD] = Categorizer.OPT_METHOD_MULTIPLE

		flag = False

		#if self.__prev_method != method or self.__prev_cutoff != threshold:
		flag = True


		'''
		if self.__prev_method != method:
			flag = True
		else:
			if method == 1: # multiple
				if self.__prev_cutoff != threshold:
					flag = True
		'''

		if flag:
			#print 'init cache!'
			self.__cache = {}
			self.__prev_method = method
			self.__prev_cutoff = threshold




		self.textBoxShowProgress.Label = 'Categorizing'
		self.textBoxShowProgress.Refresh()


		#if cpu == 1:
		#	MPI = False

		gene_category = None
		if not MPI:
			gene_category = Categorizer.process(cat_def, user_genes, org_goid_dict, option, cache = self.__cache)
		else:
			# MPI
			gene_category = Categorizer.processMPI(cat_def, user_genes, org_goid_dict, option, cache = self.__cache)
		
		# gene_category
		#     key = gene
		#     value = { category: score }
		
		print 'Gene # = ', len(gene_category)


		#-------------------------
		if gene_category is None:
			# terminate thread
			return None, None, None
		#-------------------------

		self.__putCache(gene_category)

		ref_category = {}
		if len(ref_genes)>0:
			option[Categorizer.OPT_BASAL_COUNT] = len(user_genes) # for progressbar
			
			print 'Ref gene # = ', len(ref_genes)

			if not MPI:
				ref_category = Categorizer.process(cat_def, ref_genes, org_goid_dict, option, cache = self.__cache)
			else:
				ref_category = Categorizer.processMPI(cat_def, ref_genes, org_goid_dict, option, cache = self.__cache)



			#-------------------------
			if ref_category is None:
				# terminate thread
				return None, None, None
			#-------------------------

			self.__putCache(ref_category)


		self.textBoxShowProgress.Label = 'Done..'
		self.textBoxShowProgress.Refresh()
		
		
		# for logging
		#self.__dump(gene_category, 'r:/a.txt')
		#self.__dump(ref_category, 'r:/r.txt')
		
		

		return all_categories, gene_category, ref_category

	def __dump(self, gene_category, fname):
		f=open(fname, 'w')
		
		genes = gene_category.keys()
		genes.sort()
		
		for g in genes:
			txt = [g]
			for c in gene_category[g]:
				v = gene_category[g][c]
				txt.append(c+'('+str(v)+')')
			f.write(','.join(txt)+'\n')
		f.close()

	def __putCache(self, gene_category):
		# save cache
		for k in gene_category.keys():
			if not self.__cache.has_key(k):
				self.__cache[k] = copy.deepcopy(gene_category[k])



	def OnButton1Button(self, event):

		self.__def_filename = './data/biological_processes.txt'
		self.__def_filename_flag = True

		self.__annot_filename = './data/example_gene_association.fb'
		self.__annot_filename_flag = True

		self.textCtrlUserGenes.Value = '''FBgn0036372
FBgn0010282
FBgn0036208
FBgn0035404
FBgn0027291
FBgn0260635
FBgn0015323
FBgn0051719
FBgn0035526
FBgn0035333
FBgn0030559
FBgn0039059
FBgn0036205
FBgn0036112'''



		self.textCtrlRefGenes.Value = '''
FBgn0031023
FBgn0262562
FBgn0030038
FBgn0029502
FBgn0037149
FBgn0037131
FBgn0037220
FBgn0259227
FBgn0035797
FBgn0032335
FBgn0020309
FBgn0035512
FBgn0034406
FBgn0250846
FBgn0029965
FBgn0261873
FBgn0031161
FBgn0034647
FBgn0030477
FBgn0029795
FBgn0033442
FBgn0003360
FBgn0036551
FBgn0038857
FBgn0260012
FBgn0031148
FBgn0024245
FBgn0262866
FBgn0261113
FBgn0041629
FBgn0262975
FBgn0032597
FBgn0262508
FBgn0013467
FBgn0026199
FBgn0037973
FBgn0261016
FBgn0033427
FBgn0035398
FBgn0030485
FBgn0011224
FBgn0037336
FBgn0086657
'''

		self.OnButtonRunButton(None)

		event.Skip()



	def OnButton2Button(self, event):

		self.__def_filename = './data/example_categories_simple.txt'
		self.__def_filename_flag = True

		self.__annot_filename = './data/example_gene_association.fb'
		self.__annot_filename_flag = True

		ref_file = './data/reliable_HD_mod_name.txt'
		lst = self.__loadGenes(ref_file)
		self.textCtrlUserGenes.Value = '\n'.join(lst)

		ref_file = './data/reliable_HD_ref_name.txt'
		lst = self.__loadGenes(ref_file)
		self.textCtrlRefGenes.Value = '\n'.join(lst)




		self.OnButtonRunButton(None)

		event.Skip()