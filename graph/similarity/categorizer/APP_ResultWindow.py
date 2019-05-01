# -*- coding: ms949 -*-

import wx
import MyUtil
import Categorizer
import pylab as plt
import matrix_plot
import random
import math
import sys
import os
import matplotlib
import shutil

PIE_CHART_WIDTH = 400
PIE_CHART_HEIGHT = 400
HEATMAP_WIDTH = 160
HEATMAP_HEIGHT = 380

CACHE = './temp'

WHITE = wx.Colour ( 255,255,255 )

def create(parent):
	return ResultWindow(parent)

[wxID_RESULTWINDOW, wxID_RESULTWINDOWBUTTONSAVERESULT, 
 wxID_RESULTWINDOWSTATICTEXT50,
 wxID_RESULTWINDOWLISTCTRL1, wxID_RESULTWINDOWLISTCTRLCATEGORY, 
 wxID_RESULTWINDOWSLIDERENRICHMENT, wxID_RESULTWINDOWSTATICBITMAP1, 
 wxID_RESULTWINDOWSTATICBITMAP2, wxID_RESULTWINDOWSTATICLINE1, 
 wxID_RESULTWINDOWSTATICLINE2, wxID_RESULTWINDOWSTATICTEXT1, 
 wxID_RESULTWINDOWSTATICTEXT2, wxID_RESULTWINDOWSTATICTEXT3, 
 wxID_RESULTWINDOWSTATICTEXT4, wxID_RESULTWINDOWSTATICTEXT5, 
 wxID_RESULTWINDOWTEXTCTRL1, wxID_BUTTONREDRAW
] = [wx.NewId() for _init_ctrls in range(17)]







DEFAULT_LOWER_BOUND = -3


class ResultWindow(wx.Frame):


	gene_category = None
	p_values = None
	pie_chart = None
	all_categories = None
	rid = None

	tmp_png_file = None
	tmp_data_file = None
	tmp_bar_file = None
	tmp_pie = None

	def _init_ctrls(self, prnt):

		wx.Frame.__init__(self, id=wxID_RESULTWINDOW, name='ResultWindow',
			  parent=prnt, pos=wx.Point(-1, -1), size=wx.Size(1500, 500),
			  style= wx.DEFAULT_FRAME_STYLE ^ wx.RESIZE_BORDER ^ wx.MAXIMIZE_BOX , title='Result Window')
		self.SetClientSize(wx.Size(1130, 450))
		self.SetBackgroundColour(WHITE)
		self.Bind(wx.EVT_CLOSE, self.OnCloseWindow)
		# pie chart

		self.staticBitmap_Pie = wx.StaticBitmap(bitmap=wx.NullBitmap,
			  id=wxID_RESULTWINDOWSTATICBITMAP1, name='staticBitmap1',
			  parent=self, pos=wx.Point(5, 40), size=wx.Size(408, 400),
			  style=0)

		self.staticText1 = wx.StaticText(id=wxID_RESULTWINDOWSTATICTEXT1,
			  label='Category statistics', name='staticText1', parent=self,
			  pos=wx.Point(150, 8), size=wx.Size(122, 13), style=0)



		self.listCtrl1 = wx.ListCtrl(id=wxID_RESULTWINDOWLISTCTRL1,
			  name='listCtrl1', parent=self, pos=wx.Point(410, 32),
			  size=wx.Size(140, 392), style=wx.LC_REPORT |wx.BORDER_SUNKEN)

		self.staticLine1 = wx.StaticLine(id=wxID_RESULTWINDOWSTATICLINE1,
			  name='staticLine1', parent=self, pos=wx.Point(560, 16),
			  size=wx.Size(7, 400), style=0)





		# categories
		self.listCtrlCategory = wx.ListCtrl(id=wxID_RESULTWINDOWLISTCTRLCATEGORY,
			  name='listCtrlCategory', parent=self, pos=wx.Point(570, 32),
			  size=wx.Size(320, 392), style=wx.LC_REPORT|wx.BORDER_SUNKEN)



		self.staticLine2 = wx.StaticLine(id=wxID_RESULTWINDOWSTATICLINE2,
			  name='staticLine2', parent=self, pos=wx.Point(895, 16),
			  size=wx.Size(7, 400), style=0)

		self.staticText3 = wx.StaticText(id=wxID_RESULTWINDOWSTATICTEXT3,
			  label='Categorization results', name='staticText3', parent=self,
			  pos=wx.Point(600, 8), size=wx.Size(106, 13), style=0)

		# enrichment

		self.staticText2 = wx.StaticText(id=wxID_RESULTWINDOWSTATICTEXT2,
			  label='Enrichment analysis result', name='staticText2', parent=self,
			  pos=wx.Point(920, 8), size=wx.Size(170, 13), style=0)

		# heat map
		self.staticBitmap_Enrichment = wx.StaticBitmap(bitmap=wx.NullBitmap,
			  id=wxID_RESULTWINDOWSTATICBITMAP2, name='staticBitmap2',
			  parent=self, pos=wx.Point(905, 32), size=wx.Size(336, 392),
			  style=0)

		# bar
		self.staticBitmap_EnrichmentBar = wx.StaticBitmap(bitmap=wx.NullBitmap,
			  id=wxID_RESULTWINDOWSTATICBITMAP2, name='staticBitmap2',
			  parent=self, pos=wx.Point(1090, 60), size=wx.Size(50, 100),
			  style=wx.SL_AUTOTICKS | wx.SL_LABELS)

		self.staticText40 = wx.StaticText(id=wxID_RESULTWINDOWSTATICTEXT4,
			  label='log10(p-value)', name='staticTextBarExp', parent=self,
			  pos=wx.Point(1080,  45), size=wx.Size(80, 13), style=0)



		# slider
		p=1035
		self.sliderEnrichment = wx.Slider(id=wxID_RESULTWINDOWSLIDERENRICHMENT,
			  maxValue=-1, minValue=-15, name='sliderEnrichment', parent=self,
			  pos=wx.Point(1090, 260), size=wx.Size(30, 100),
			  style=wx.SL_VERTICAL, value=DEFAULT_LOWER_BOUND)
		self.sliderEnrichment.SetToolTip( wx.ToolTip( str( self.sliderEnrichment.GetValue() ) ) )
		#self.sliderEnrichment.SetBackgroundColour('yellow')


		# EVT_SCROLL_CHANGE <- windows only
		self.sliderEnrichment.Bind(wx.EVT_SCROLL,
			  self.OnSliderEnrichmentCommandScroll,
			  id=wxID_RESULTWINDOWSLIDERENRICHMENT)



		self.staticText41 = wx.StaticText(id=wxID_RESULTWINDOWSTATICTEXT4,
			  label='1e-15', name='staticTextSliderMax', parent=self,
			  pos=wx.Point(1130,
			  260), size=wx.Size(40, 15), style=0)

		self.staticText51 = wx.StaticText(id=wxID_RESULTWINDOWSTATICTEXT5,
			  label='1e-1', name='staticTextSliderMin', parent=self,
			  pos=wx.Point(1130,
			  350), size=wx.Size(40, 15), style=0)




		# redraw button

		self.buttonRedraw = wx.Button(id=wxID_BUTTONREDRAW,
			  label='Redraw', name='buttonRedraw', parent=self,
			  pos=wx.Point(1080,
			  370), size=wx.Size(70, 33), style=0)
		self.buttonRedraw.Bind(wx.EVT_BUTTON, self.OnButtonRedraw,
			  id=wxID_BUTTONREDRAW)
		#self.buttonRedraw.SetBackgroundColour( 'yellow' )


		# menu
		self.menu_bar = wx.MenuBar()
		self.menu_menu = wx.Menu()
		msave = self.menu_menu.Append( wx.ID_SAVEAS, 'Save results')
		mclose = self.menu_menu.Append( wx.ID_EXIT, 'Close')
		self.menu_bar.Append( self.menu_menu, 'Menu')
		self.SetMenuBar(self.menu_bar)

		self.Bind( wx.EVT_MENU, self.OnUpdateMenuSave, msave)
		self.Bind( wx.EVT_MENU, self.OnUpdateMenuClose, mclose)



		# icon
		path = os.path.abspath("./cat.ico")
		icon = wx.Icon(path, wx.BITMAP_TYPE_ICO)
		self.SetIcon(icon)




	def OnUpdateMenuSave(self, event):

		evt_id = event.GetId()

		if evt_id == wx.ID_SAVEAS:
			self.__saveData()

		else:
			event.Skip()

	def __saveData(self):

		dialog = wx.DirDialog(None,
							  "Choose a folder to save results",
							  style = wx.DD_DEFAULT_STYLE | wx.DD_NEW_DIR_BUTTON)

		if dialog.ShowModal() == wx.ID_OK:
			path = dialog.GetPath()
			self.save(path)
			result = wx.MessageBox("Saved in " + path, style = wx.ICON_INFORMATION | wx.OK )

		dialog.Destroy()

	def save(self, path):



		self.__copyFile(self.tmp_pie, path)

		self.__copyFile(self.tmp_png_file, path)
		self.__copyFile(self.tmp_bar_file, path)
		self.__copyFile(self.tmp_data_file, path)

		self.__saveStatFile( path + '/stat.txt' )


		self.__saveCategoryFile( path + '/categories.txt')

	def __saveStatFile(self, fname):

		f = open(fname, 'w')
		f.write('Category\tOccurrence\n')

		for cat in self.all_categories:
			v = self.pie_chart[cat]

			f.write(cat + '\t' + str(v).strip() + '\n' )

		f.close()

	def __saveCategoryFile(self, fname):

		f = open(fname, 'w')


		data = []
		keys = self.gene_category.keys()
		keys.sort()

		for g in keys:

			gene = g
			txt = []


			for cat in MyUtil.sortDict( self.gene_category[g], descending_order=True ):

				ttt = cat+'('+str( round( self.gene_category[g][cat], 2)).strip()+')'
				if cat == Categorizer.CATEGORY_NO_ANNOTATION:
					ttt = '-'

				txt.append(ttt)

			sss = gene + '\t' + ', '.join(txt)
			f.write(sss+'\n')

		f.close()


	def __copyFile(self, your_file, dest_folder):

		if os.path.exists(your_file):

			base = os.path.basename(your_file)
			target = base.replace( self.rid + '__', '')

			if target.find('_bar.png')>0:
				target = 'heatmap_bar.png'

			shutil.copy(your_file, dest_folder + '/' + target)


	def close(self):
		self.delete( self.tmp_bar_file )
		self.delete( self.tmp_data_file )
		self.delete( self.tmp_pie )
		self.delete( self.tmp_png_file )

	def delete(self, fname):

		try:
			if os.path.exists(fname):
				os.remove(fname)

		except:
			pass


	def OnCloseWindow(self, event):
		evt_id = event.GetId()

		if self.ask("Do you want to close this window?"):
			self.close()

			self.MakeModal(False)

			self.Destroy()


	def OnUpdateMenuClose(self, event):

		evt_id = event.GetId()


		if evt_id == wx.ID_EXIT:
			if self.ask("Do you want to close this window?"):
				self.close()

				self.MakeModal(False)

				self.Destroy()

		else:
			event.Skip()


	def ask(self, msg):

		result = wx.MessageBox(msg, style = wx.CENTER | wx.ICON_QUESTION | wx.YES_NO )
		if result == wx.YES:
			return True
		else:
			return False


	def __init__(self, parent):

		self._init_ctrls(parent)

		self.gene_category = None
		self.p_values = None
		self.pie_chart = None
		self.all_categories = None
		self.rid = MyUtil.getRandomString(5)

		self.tmp_png_file = CACHE + '/' + self.rid + '__heatmap.png'
		self.tmp_data_file = CACHE + '/' + self.rid + '__heatmap.txt'
		self.tmp_bar_file = self.tmp_png_file  + '_bar.png'
		self.tmp_pie = CACHE + '/' + self.rid + '__piechart.png'


	def process(self, all_categories, gene_category, pie_chart, p_values):

		self.all_categories = all_categories
		self.gene_category = gene_category
		self.pie_chart = pie_chart
		self.p_values = p_values


		self.__showPieChart(all_categories, pie_chart)

		try:
			self.__showPieChart(all_categories, pie_chart)
		except:
			wx.MessageBox("Error in showing categorization result.", caption="Error", style=wx.OK|wx.CENTER|wx.ICON_ERROR, parent=None, x=-1, y=-1)



		self.__showCategories(all_categories, gene_category)

		if p_values is not None:

			try:
				self.__showHeatMap(all_categories, p_values)
			except:
				wx.MessageBox("Error in showing enrichment result.", caption="Error", style=wx.OK|wx.CENTER|wx.ICON_ERROR, parent=None, x=-1, y=-1)



			s = self.GetSize()
			s[0] = self.buttonRedraw.GetPosition()[0] + self.buttonRedraw.GetSize()[0] + 50
			self.SetSize( s )

		else:

			s = self.GetSize()
			s[0] = self.staticLine2.GetPosition()[0] + self.staticLine2.GetSize()[0] + 10
			self.SetSize( s)

	def __showHeatMap(self, all_categories, p_values):


		self.__saveData4HeatMap(all_categories, p_values, self.tmp_data_file)


		min_value = self.sliderEnrichment.Value

		opt = [
			None,
			self.tmp_data_file,
			'"-o:' + self.tmp_png_file+'"',
			'-rotation_y_axis',
			'-max:0',
			'-min:' + str(min_value).strip(),
			'-blank:black',
			]

		o = matrix_plot.getOption( opt )

		matrix_plot.run(o)



		img = wx.Bitmap(self.tmp_png_file, wx.BITMAP_TYPE_PNG)
		x, y = img.GetSize()

		w, h = self.__getOptimalBitmapSizeOfHeatMap(x, y)

		img2 = self.scale_bitmap( img, w, h)
		self.staticBitmap_Enrichment.SetBitmap( img2 )





		imgbar = wx.Bitmap(self.tmp_bar_file, wx.BITMAP_TYPE_PNG)
		x, y = imgbar.GetSize()

		h = 180
		w = x * h /y


		imgbar2 = self.scale_bitmap(imgbar, w, h)
		self.staticBitmap_EnrichmentBar.SetBitmap(imgbar2)


		self.Refresh()




	def __saveData4HeatMap(self, all_categories, p_values, data_file):

		f=open(data_file,'w')

		f.write('Category\tp-value\n')
		for cat in all_categories:
			
			
			if cat == Categorizer.CATEGORY_NO_ANNOTATION:
				continue
			
			print cat, p_values[cat]


			y = str(p_values[cat]).strip()
			if y == 'nan':

				p_values[cat] = 1.0



			s = p_values[cat]
			if s < sys.float_info.min:
				s = sys.float_info.min

			log_p = str( math.log10( s ) ).strip()
			f.write( cat + '\t' + log_p + '\n')



		f.close()

	def __showPieChart(self, all_categories, pie_chart):



		alpha=0.3

		rgb = matplotlib.colors.ColorConverter()

		color_map = [  'w', \
					  rgb.to_rgba( (0.0, 0.0, 1.0), alpha=alpha), # b
					  rgb.to_rgba( (0.0, 0.5, 0.0), alpha=alpha), # g
					  rgb.to_rgba( (1.0, 0.0, 0.0), alpha=alpha), # r
					  rgb.to_rgba( (0.0, 0.75, 0.75), alpha=alpha), # c
					  rgb.to_rgba( (0.75, 0, 0.75), alpha=alpha), # m
					  rgb.to_rgba( (0.75, 0.75, 0), alpha=alpha), # y
					  rgb.to_rgba( (0.0, 0.0, 0.0), alpha=alpha), # k
					  'b', 'g', 'r', 'c', 'm', 'k', 'y'
					  ]




		self.listCtrl1.ClearAll()
		self.listCtrl1.InsertColumn(0, 'Category')
		self.listCtrl1.InsertColumn(1, '#')
		self.listCtrl1.SetColumnWidth(0, 110)
		self.listCtrl1.SetColumnWidth(1, 50)



		number = []
		label = []

		total = 0.0
		for cat in pie_chart:
			total += pie_chart[cat]


		for cat in all_categories:

			per = 0.0
			if total != 0.0:
				per = float( pie_chart[cat]) / float(total) * 100.0

			perstr = "%f.1" % per

			lgd = cat + '(' + perstr + ' %)'

			label.append(cat)
			number.append( pie_chart[cat] )

			self.listCtrl1.Append( (cat, str(int(pie_chart[cat] )) ))



		self.listCtrl1.Refresh()



		plt.close()



		fig = plt.figure(1, figsize=(4,5))

		ax = plt.subplot(111)

		x = ax.pie(number, labels = label, colors = color_map)

		handles, labels = ax.get_legend_handles_labels()
		ax.legend( handles, label, loc = 'best',  bbox_to_anchor =(1.2, 0.5), shadow=True )


		fig.tight_layout()
		plt.tight_layout()

		plt.savefig(self.tmp_pie, dpi=300, bbox_inches='tight')
		plt.clf()

		img = wx.Bitmap(self.tmp_pie, wx.BITMAP_TYPE_PNG)
		[x,y] = img.GetSize()



		#----------------------------------------
		[w, h ] = self.__getOptimalBitmapSizeOfPieChart(x,y)



		img2 = self.scale_bitmap( img, w, h)

		self.staticBitmap_Pie.SetBitmap( img2 )
		
		


	def __getOptimalBitmapSizeOfPieChart(self, x, y):

		w = h = 0

		if float(x)/float(y) > float(PIE_CHART_WIDTH)/float(PIE_CHART_HEIGHT):


			w = PIE_CHART_WIDTH
			h = w * y / x

		else:

			h = PIE_CHART_HEIGHT
			w = x * h / y

		return w, h


	def __getOptimalBitmapSizeOfHeatMap(self, x, y):

		w = h = 0

		if float(x)/float(y) > float(HEATMAP_WIDTH)/float(HEATMAP_HEIGHT):


			w = HEATMAP_WIDTH
			h = w * y / x

		else:

			h = HEATMAP_HEIGHT
			w = x * h / y

		return w, h

	def scale_bitmap(self, bitmap, width, height):
		image = wx.ImageFromBitmap(bitmap)
		image = image.Scale(width, height, wx.IMAGE_QUALITY_HIGH)
		result = wx.BitmapFromImage(image)
		return result


	def __showCategories(self, all_categories, gene_category):


		self.listCtrlCategory.ClearAll()
		self.listCtrlCategory.InsertColumn(0, 'Gene')
		self.listCtrlCategory.InsertColumn(1, 'Category')
		self.listCtrlCategory.SetColumnWidth(0, 110)
		self.listCtrlCategory.SetColumnWidth(1, 220)

		data = []
		keys = gene_category.keys()
		keys.sort()

		for g in keys:

			gene = g
			txt = []


			for cat in MyUtil.sortDict( gene_category[g], descending_order=True ):

				ttt = cat+'('+str( round( gene_category[g][cat], 2)).strip()+')'
				if cat == Categorizer.CATEGORY_NO_ANNOTATION:
					ttt = '-'


				txt.append(ttt)


			self.listCtrlCategory.Append(
				( g, ', '.join(txt) )
			)

			self.listCtrlCategory.Refresh()

	def OnButtonSaveResultButton(self, event):
		event.Skip()




	def OnSliderEnrichmentScrollThumbrelease(self, event):
		event.Skip()

	def OnSliderEnrichmentScroll(self, event):
		self.sliderEnrichment.SetToolTip( wx.ToolTip( str( self.sliderEnrichment.GetValue() ) ) )

	def OnSliderEnrichmentCommandScrollThumbrelease(self, event):
		self.sliderEnrichment.SetToolTip( wx.ToolTip( str( self.sliderEnrichment.GetValue() ) ) )


	def OnButtonRedraw(self, event):
		self.buttonRedraw.Enabled = False
		self.__redrawEnrichmentHeatmap()
		self.buttonRedraw.Enabled = True

	def OnSliderEnrichmentCommandScroll(self, event):

		self.sliderEnrichment.SetToolTip( wx.ToolTip( str( self.sliderEnrichment.GetValue() ) ) )

	def __redrawEnrichmentHeatmap(self):
		self.__showHeatMap(self.all_categories, self.p_values)