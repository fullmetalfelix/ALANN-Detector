import tkinter as tk
from tkinter import ttk
from tkinter import filedialog # needed so we can open a file explorer window when looking for the .gds/.bmp/etc files
import customtkinter

import numpy
import itertools as it
from enum import IntEnum

from scanner import Sample, SampleCheck, Scanner, SPM

import PIL
from PIL import Image, ImageTk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib import pyplot as plt

import GDSConverter # some custom classes/functions for importing and converting files (gds specifically atm) to vector coordinates for the tip



customtkinter.set_appearance_mode("dark")  # Modes: system (default), light, dark
customtkinter.set_default_color_theme("dark-blue")  # Themes: blue (default), dark-blue, green




class PhysicalSizes(IntEnum):
	i10nm = 10
	i50nm = 50
	i100nm = 100
	i200nm = 200
	i500nm = 500
	i750nm = 750
	i1um = 1000
	i2um = 2000
	i5um = 5000
	i7um = 7500
	i10um = 10000
	i12um = 12000

def resizing(frame, rows, columns, weight=1):
	#function that takes in a list of rows and columns that should be dynamically resized as the window is resized
	for i in rows:
		frame.grid_rowconfigure(i, weight=1)
	for i in columns:
		frame.grid_columnconfigure(i, weight=1)


class CanvasLine(object):

	def __init__(self, name, points, **kwargs):

		# params is a numpy matrix Nx2 physical space coordinates
		# where N is the number of points
		self.points = points
		self.name = name

		self.options = kwargs


	def render(self, canvas):

		
		last = None
		for i in range(self.points.shape[0]):

			# compute canvas px positions
			q = canvas.physical_to_canvas(self.points[i])

			# draw the line
			if i > 0:
				canvas.create_line(last[0], last[1], q[0], q[1], **self.options)

			last = q

class CanvasPoint(object):

	def __init__(self, name, coords, pxsize, **kwargs):

		self.name = name

		self.coords = coords
		self.pxsize = pxsize

		self.options = kwargs


	def render(self, canvas):

		p = canvas.physical_to_canvas(self.coords)
		canvas.create_oval(p[0]-self.pxsize, p[1]-self.pxsize, p[0]+self.pxsize, p[1]+self.pxsize, **self.options)

class CanvasSPM(object):

	def __init__(self, name, spm):

		self.name = name
		self.spm = spm
		

	def render(self, canvas):

		spm = self.spm

		# get the vertexes of this spm in canvas px coordinates
		# ASSUMPTIONS (DEBUG!!!):
		# 	angle is 0
		# 	slow scan is from bottom to top (positive y axis in physical space)
		# 	fast scan is in the positive x physical axis


		# determine which part of the picture to draw
		# these are the canvas corners in physical space
		y0 = canvas.corners[3,1]
		ym = canvas.corners[0,1]
		x0 = canvas.corners[0,0]
		xm = canvas.corners[1,0]

		#print("canvas corners:",[x0,y0],[xm,ym])

		# if both corners of an edge are on the same side of the canvas, the image is out
		spm_x0 = spm.frame_corners[0,0]
		spm_xm = spm.frame_corners[1,0]

		frame_x0 = numpy.max([x0,spm_x0])
		frame_xm = numpy.min([xm,spm_xm])

		#print("frame x",frame_x0,frame_xm,x0,xm)

		if frame_xm < x0 or frame_x0 > xm:
			#print("spm is out of canvas (x)")
			return None

		spm_y0 = spm.frame_corners[0,1]
		spm_ym = spm.frame_corners[2,1]

		frame_y0 = numpy.max([y0,spm_y0])
		frame_ym = numpy.min([ym,spm_ym])

		if frame_ym < y0 or frame_y0 > ym:
			#print("spm is out of canvas (y)")
			return None

		# code here => there is some overlap between spm and canvas
		#print("frame boundaries on spm (x):",[spm_x0,frame_x0],[spm_xm,frame_xm])
		#print("frame boundaries on spm (y):",[spm_y0,frame_y0],[spm_ym,frame_ym])



		# convert height values to color
		# this can make the topography contrast go away quite a bit
		data = spm.data - canvas.SPM_min # also applies the shift
		data /= canvas.SPM_max/1.5
		data *= 255

		# final conversion to bytes and flip vertically
		data = data.astype(numpy.uint8)
		data = numpy.flip(data, axis=0)

		# create the PIL image object from data
		pic = Image.fromarray(data)
		rot = pic.rotate(spm.angle, expand=True)
		# make a rotation mask
		mask = numpy.zeros(data.shape,dtype=numpy.uint8)
		mask += 255
		mask = Image.fromarray(mask)
		mask = mask.rotate(spm.angle, expand=True)

		# this is completely white-transparent image to blend with rot using mask
		bgim = numpy.zeros((data.shape[0],data.shape[1],4),dtype=numpy.uint8)
		bgim[:,:,0] = bgim[:,:,1] = bgim[:,:,2] = 255
		bgim = Image.fromarray(bgim, mode="RGBA")
		bgim = bgim.rotate(spm.angle, expand=True)

		rotm = Image.composite(rot, bgim, mask)

		# crop the image
		# where is frame_x0 in spm pixel coordinates?
		frame_px_x0 = int(numpy.floor((frame_x0 - spm_x0) / spm.pixelSize[0]))
		frame_px_xm = int(numpy.ceil((frame_xm-spm_x0) / spm.pixelSize[0]))
		if frame_px_xm == 0: frame_px_xm = 1

		#print("frame pixel coords (x)",frame_px_x0,frame_px_xm)
		#data = data[:,frame_px_x0:frame_px_xm+1]

		frame_px_y0 = int(numpy.floor((frame_y0 - spm_y0) / spm.pixelSize[1]))
		frame_px_ym = int(numpy.ceil((frame_ym-spm_y0) / spm.pixelSize[1]))
		if frame_px_ym == 0: frame_px_ym = 1

		#print("frame pixel coords (y)",frame_px_y0,frame_px_ym)
		#data = data[frame_px_y0:frame_px_ym+1]

		#print("data stats",numpy.mean(spm.data),numpy.min(spm.data),numpy.max(spm.data))

		# perform the crop
		cropbox = (frame_px_x0, rotm.size[1]-frame_px_ym, frame_px_xm, rotm.size[1]-frame_px_y0)
		#print("cropping",rotm.size, cropbox)
		pic = rotm.crop(cropbox)
		

		# resample to match canvas resolution

		# we have to make the spm pixels the same size as the canvas pixels
		# canvas pixel size is 1 / self.canvas_res
		# spm pixel size is spm.pixelSize (x,y components)

		trgPXsize = numpy.asarray([1,1]) / canvas.resolution
		curPXsize = spm.pixelSize
		scaling = curPXsize / trgPXsize
		newsize = numpy.ceil(numpy.asarray([pic.size[0],pic.size[1]]) * scaling)
		newsize = newsize.astype(numpy.uint32)
		method = Image.Resampling.BICUBIC
		if scaling[0] < 1 and scaling[1] < 1:
			method = Image.Resampling.LANCZOS
		#print(trgPXsize,curPXsize,scaling,"--",pic.size, newsize)

		pic = pic.resize(newsize, resample=method)
		tkpic = ImageTk.PhotoImage(image=pic)

		self._crop = tkpic

		p = numpy.asarray([frame_x0, frame_y0], dtype=numpy.float64)
		c = canvas.physical_to_canvas(p)
		#print("canvas placement:",p,c)
		canvas.create_image(c[0],c[1], image=tkpic, anchor="sw")
		#print("the spm is now {}w x {}h [nm]".format(pic.size[0]/self.canvas_res, pic.size[1]/self.canvas_res))
		
class CanvasCrossHair(object):

	def __init__(self, name, position, **kwargs):

		self.name = name

		# in physical space
		self.position = position

		self.options = kwargs


	def render(self, canvas):

		tip = self.position
		ctip = canvas.physical_to_canvas(tip)
		
		canvas.create_line(ctip[0], ctip[1]-8, ctip[0], ctip[1]-2, **self.options)
		canvas.create_line(ctip[0], ctip[1]+8, ctip[0], ctip[1]+2, **self.options)

		canvas.create_line(ctip[0]-8, ctip[1], ctip[0]-2, ctip[1], **self.options)
		canvas.create_line(ctip[0]+8, ctip[1], ctip[0]+2, ctip[1], **self.options)


class PhysicalCanvas(tk.Canvas):




	def __init__(self, parent, **kwargs):

		tk.Canvas.__init__(self, parent, **kwargs)
		self.configure(**kwargs)

		self.parent = parent

		# center of the canvas in physical space
		self.center = numpy.asarray([0,0], dtype=numpy.float64)

		# canvas resolution in px/nm
		self.resolution = 128
		
		# canvas widget size in pixels - will be set by resize
		self.size = numpy.zeros(2, dtype=numpy.int32)

		# canvas corner positions in physical space - order is ABCD clockwise A = top-left = canvas 0,0
		self.corners = numpy.zeros((4,2), dtype=numpy.float64)


		self._axisflipper = numpy.asarray([1,-1], dtype=numpy.float64)


		self._stackPoints = []
		self._stackLines = []
		self._stackSPM = []

		self.hasFocus = False


		self.variables = {

			'resolution': 	{"object": tk.StringVar(value="..."), "value": None},
			'mousepos': 	{"object": tk.StringVar(value="..."), "value": None}
		}


		self.callbacks = {
			'click': [],

		}


		#self.bind("<FocusOut>", self.lose_focus)
		self.bind("<1>", self._onclick)

		self.bind("<Configure>", self._resize)
		self.bind('<Motion>', self._onMouseMove)
		self.bind('<Leave>', self._onMouseOut)

		self.bind("<q>", lambda e: self.zoom(True))
		self.bind("<e>", lambda e: self.zoom(False))

		self.bind("<w>", lambda e: self.move([0,-1]))
		self.bind("<a>", lambda e: self.move([-1,0]))
		self.bind("<s>", lambda e: self.move([0,1]))
		self.bind("<d>", lambda e: self.move([1,0]))


	### FOCUS EVENTS ### ##############################################

	def _onclick(self, event):

		self.give_focus()
		#print("canvas click")

		for cb in self.callbacks['click']:
			cb(event)

		self._onMouseMove(event)

	def give_focus(self):
		#print(self,"get focus")
		self.hasFocus = True
		self.focus_set()
		self.configure(background="white")

	def lose_focus(self):
		#print(self,"lost focus")
		self.hasFocus = False
		self.configure(background="gray")

	def _onMouseMove(self, event):
		
		x, y = event.x, event.y
		c = numpy.asarray([x,y])
		p = self.canvas_to_physical(c)

		self.variables['mousepos']['value'] = p

		magn0, units0, dummy = self.physical_to_approximate(p[0],3)
		magn1, units1, dummy = self.physical_to_approximate(p[1],3)

		self.variables['mousepos']['object'].set(
			"x:{:+.3f} {}, y:{:+.3f} {}".format(magn0,units0, magn1,units1)
		)


	def _onMouseOut(self, event):

		self.variables['mousepos']['object'].set("N/A")
		self.variables['mousepos']['value'] = None

	###################################################################

	### CANVAS CONTROLS ### ###########################################

	def zoom(self, inc=False):

		if inc:
			if self.resolution < 512:
				self.resolution *= 2
				self._resize(None)
		else:
			if self.resolution > 5.0e-07:
				self.resolution /= 2
				self._resize(None)


	def move(self, direction):

		step = 0.1 * self.size / self.resolution
		self.center += numpy.asarray(direction) * self._axisflipper * step
		self._resize(None)
		

	###################################################################

	### POSITIONING ### ###############################################

	## set the physical space center of the canvas and the resolution (if given)
	def setSpace(self, center=None, resolution=None):

		upd = False

		if center is not None:
			self.center = numpy.asarray(center, dtype=numpy.float64)
			upd = True

		if resolution:
			self.resolution = resolution
			upd = True


		if upd:
			self._resize(None)


	## converts coordinates from physical space into canvas pixel space
	def physical_to_canvas(self, point):

		v = self.size * 0.5
		v += self._axisflipper * (point - self.center) * self.resolution
		return v

	## converts coordinates from pixel space into physical space
	def canvas_to_physical(self, pxpoint):

		v = self._axisflipper * pxpoint
		v-= self._axisflipper * self.size*0.5
		v/= self.resolution
		v+= self.center

		return v

	def physical_to_approximate(self, length, decimals=0):

		# the input length must be in nm

		# round the nm size to a convenient number
		units = "nm"
		magn = length
		magn_nm = length

		rounder = numpy.power(10,decimals)
		prefix = 1

		if numpy.abs(length) > 1000000:
			
			units = "mm" # units become um
			magn /= 1000000
			prefix = 1000000
		
		elif numpy.abs(length) > 1000:
			
			units = "μm" # units become um
			magn /= 1000
			prefix = 1000

		elif numpy.abs(length) < 0.1:

			units = "pm" # units become pico
			prefix = 0.001
			magn *= 1000

		elif numpy.abs(length) < 1:

			units = "Å" # units become angs
			prefix = 0.1
			magn *= 10


		else:
			
			units = "nm"
			prefix = 1
			
		# rounds to the requested decimal
		magn = numpy.round(magn*rounder) / rounder
		magn_nm = magn*prefix

		return magn, units, magn_nm

	###################################################################





	def _compute_corners(self):

		# get the widget shape
		self.size[0] = self.winfo_width()
		self.size[1] = self.winfo_height()

		# compute the canvas corner positions in physical space
		p = numpy.zeros(2)
		self.corners[0] = self.canvas_to_physical(p)
		p[0] = self.size[0]
		self.corners[1] = self.canvas_to_physical(p)
		p[1] = self.size[1]
		self.corners[2] = self.canvas_to_physical(p)
		p[0] = 0
		self.corners[3] = self.canvas_to_physical(p)


	def _resize(self, event):

		self._compute_corners()
		
		# set resolution variable
		self.variables['resolution']['value'] = self.resolution
		if self.resolution >= 1:
			self.variables['resolution']['object'].set("{} px/nm".format(self.resolution))
		else:
			self.variables['resolution']['object'].set("{}⁻¹ px/nm".format(1.0/self.resolution))

		self.render()




	def ClearStack(self):

		self._stackPoints = []
		self._stackLines = []
		self._stackSPM = []
		self.render()

	def AddObject(self, cobj, noRender=False):

		if isinstance(cobj, CanvasPoint):
			self._stackPoints.append(cobj)
		elif isinstance(cobj, CanvasLine) or isinstance(cobj, CanvasCrossHair):
			self._stackLines.append(cobj)
		elif isinstance(cobj, CanvasSPM):
			self._stackSPM.append(cobj)
		else:
			raise TypeError("Invalid canvas object")


		if not noRender:
			self.render()

	def RemoveObject(self, name, noRender=False):

		self._stackPoints = [o for o in self._stackPoints if o.name != name]
		self._stackLines = [o for o in self._stackLines if o.name != name]
		self._stackSPM = [o for o in self._stackSPM if o.name != name]

		if not noRender:
			self.render()



	def _draw_scalebar(self):

		cw = self.size[0]
		ch = self.size[1]

		barheight = 20


		barsize_px = 0.1 * cw # bar size in pixels - how many nm is that?
		barsize_nm = barsize_px / self.resolution # size in nm -> round it

		# round the nm size to a convenient number
		units = "nm"
		magn = 0

		magn, units, barsize_nm = self.physical_to_approximate(barsize_nm, 0)

		# then get the fixed pixel count
		barsize_px = numpy.round(barsize_nm * self.resolution)
		bartxt = "{} {}".format(magn, units)

		
		self.create_rectangle(cw-20-barsize_px, ch-20-barheight, cw-20, ch-20, fill="black",outline="white", width=2)
		self.create_rectangle(cw-20-2*barsize_px, ch-20-barheight+2, cw-20-barsize_px, ch-20-2, fill="white",outline="black", width=2)
		self.create_text(cw-20-barsize_px/2, ch-20-barheight/2, justify=tk.CENTER, text=bartxt, fill="white")





	def render(self):

		self.delete("all")


		# first render the SPMs

		# get the global min/max
		imgmin = float("inf")
		imgmax = float("-inf")
		for spmobj in self._stackSPM:
			spm = spmobj.spm
			m = numpy.min(spm.data)
			imgmin = min(m, imgmin)

			m = numpy.max(spm.data)
			imgmax = max(m, imgmax)

		self.SPM_min = imgmin
		self.SPM_max = imgmax

		# sort images by resolution - low res images are drawn first
		scans = sorted(self._stackSPM, key=lambda x: x.spm.pixelSize[0], reverse=True)
		

		for o in scans: o.render(self)


		# then the lines
		for o in self._stackLines: o.render(self)

		# last the points
		for o in self._stackPoints: o.render(self)



		# add the scale bar
		self._draw_scalebar()



# this is the main window which contains the different tabs and controls which is being seen
class ALANNGUI(customtkinter.CTk):


	def __init__(self, *args, **kwargs):

		customtkinter.CTk.__init__(self, *args, **kwargs)
		self.geometry("1200x800")
		self.title('ALANN')

		# this is the main container
		container = customtkinter.CTkFrame(self)
		container.pack(side="top", fill="both", expand=True)
		#container.grid(row=0, column=0, sticky="nsew")
		
		# what does this do?
		resizing(container, [1], [0])

		# vertical side menu that allows navigation to different tabs
		menu = customtkinter.CTkFrame(container)
		menu.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)


		self.menu_width=90

		self.tabInfo = {
			'nav':{
				'class': 'TabHome',
				'name': 'Navigation',
				'button': None,
				'frame': None
			},
			'path':{
				'class': 'TabLithoPath',
				'name': 'Pathing',
			}
		}

		# create the tabs and tab selector buttons
		col = 0
		for tn in self.tabInfo.keys():
			
			tab = self.tabInfo[tn]
			cmd = lambda tn=tn: self.tab_show(tn)


			tab['button'] = customtkinter.CTkButton(menu, text=tab['name'], command=cmd, width=self.menu_width, text_color_disabled="black")
			tab['button'].grid(row=0, column=col, padx=2, pady=2)
			tab['button'].configure(fg_color="#4682bd")

			frame = globals()[tab['class']](container, self)
			frame.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
			tab['frame'] = frame

			col += 1

		self.tab_show('nav')


	def tab_show(self, tabname):

		# brings forward the frame of the tab you want to see
		tab = self.tabInfo[tabname]

		for tn in self.tabInfo.keys():
			if tn != tab['class']:
				self.tabInfo[tn]['button'].configure(fg_color="#4682bd", state=tk.NORMAL)
				if self.tabInfo[tn]['frame'].canvas:
					self.tabInfo[tn]['frame'].canvas.lose_focus()

		tab['button'].configure(fg_color="#46bd64", state=tk.DISABLED)
		tab['frame'].tkraise()
		if tab['frame'].canvas:
			tab['frame'].canvas.give_focus()





# this is the main navigation/scanning panel of the GUI
class TabHome(customtkinter.CTkFrame):
	
	# max image size is 12 μm
	# more values in between are needed
	# 1200nm 4.5um ...


	def __init__(self, parent, controller):	
		
		customtkinter.CTkFrame.__init__(self, parent)
		
		self.alanngui = controller
		self._scans = []
	
		self.variables = {
			'tippos': {'object': tk.StringVar(value="..."), 'value': numpy.asarray([0,0], dtype=numpy.float64)},
		}

		self.grid_rowconfigure(0, weight=1)
		self.grid_columnconfigure(0, weight=0, minsize=400)
		self.grid_columnconfigure(1, weight=2, minsize=400)



		# and this is the map panel - for the canvas
		frame_map = customtkinter.CTkFrame(master=self, corner_radius=4)
		frame_map.grid(row=0, column=1, padx=4, pady=4, sticky="nsew")

		frame_map.grid_columnconfigure(0, weight=2)
		frame_map.grid_rowconfigure(0, weight=2)

		# canvas - this has to go first
		canvas = PhysicalCanvas(frame_map, background="white")
		canvas.grid(row=0, column=0,padx=4,pady=4, sticky="nsew")
		self.canvas = canvas



		# this should be the main control bar on the left
		frame_ctrl = customtkinter.CTkFrame(master=self, width=250, corner_radius=4)
		frame_ctrl.grid(row=0, column=0, padx=8, pady=8, sticky="nsew")
		#frame_ctrl.grid_propagate(False)
		self.frame_ctrl = frame_ctrl

		customtkinter.CTkLabel(master=frame_ctrl,text="Navigation & Mapping", text_font = ("Roboto",14)).grid(row=0, column=0, pady=4)

		frm_scan = self._init_scan_panel(frame_ctrl)
		frm_scan.grid(row=1, column=0, pady=4, padx=4,sticky="new")
		



		# canvas navigation panel
		frame_map_ctrl = self._init_nav_panel(frame_ctrl)
		frame_map_ctrl.grid(row=2, column=0, padx=4, pady=4, sticky="new")

		canvas.callbacks['click'].append(self.MoveTip)
		#canvas.bind("<Button-1>", self.canvas_onclick)


		# start with tip in 0,0
		# this is a crosshair object
		self.crosshair = CanvasCrossHair("tippos", self.variables['tippos']['value'], fill='red')
		canvas.AddObject(self.crosshair, noRender=True)

		self._onTipPosChange([0,0])



	### SUB PANELS ### ########################################

	def _init_scan_panel(self, mainframe):

		frm_scan = customtkinter.CTkFrame(master=mainframe,corner_radius=4)

		# title label
		customtkinter.CTkLabel(master=frm_scan,text="Imaging Parameters").grid(row=0, column=0, columnspan=3)
		nrow = 1

		# image px size panel

		customtkinter.CTkLabel(master=frm_scan,text="pixels:").grid(row=nrow, column=0)

		sld_px = customtkinter.CTkSlider(master=frm_scan, from_=6, to=11, number_of_steps=5, command=self.pxsize_change)
		sld_px.grid(row=nrow, column=1)
		self.sld_px = sld_px

		tvar_pxsize = tk.StringVar(value="...")
		self.tvar_pxsize = tvar_pxsize
		lbl_pxsize = customtkinter.CTkLabel(master=frm_scan,textvariable=tvar_pxsize).grid(row=nrow, column=2)
		
		sld_px.set(8)
		nrow += 1


		# img physical size

		customtkinter.CTkLabel(master=frm_scan,text="size:").grid(row=nrow, column=0)

		# from 100nm to 12um
		vals = [e for e in PhysicalSizes]
		sld_ph = customtkinter.CTkSlider(master=frm_scan, from_=0, to=len(vals)-1, number_of_steps=len(vals)-1, command=self.phsize_change)
		sld_ph.grid(row=nrow, column=1)
		self.sld_ph = sld_ph


		tvar_phsize = tk.StringVar(value="...")
		self.tvar_phsize = tvar_phsize

		customtkinter.CTkLabel(master=frm_scan,textvariable=tvar_phsize).grid(row=nrow, column=2)
		sld_ph.set(8)
		nrow += 1


		# img fast-scan angle

		customtkinter.CTkLabel(master=frm_scan,text="angle:").grid(row=nrow, column=0)
		
		sld_angle = customtkinter.CTkSlider(master=frm_scan, from_=-90, to=90, number_of_steps=180, command=self.phang_change)
		sld_angle.grid(row=nrow, column=1)
		self.sld_angle = sld_angle

		tvar_phang = tk.StringVar(value="...")
		self.tvar_phang = tvar_phang
		customtkinter.CTkLabel(master=frm_scan,textvariable=tvar_phang).grid(row=nrow, column=2)

		
		sld_angle.set(0)
		nrow += 1

		bt_scan = customtkinter.CTkButton(master=frm_scan, text="SCAN", command=self.scan_click)
		bt_scan.grid(row=nrow, column=1,pady=4)

		return frm_scan

	def _init_nav_panel(self, mainframe):

		frame_map_ctrl = customtkinter.CTkFrame(master=mainframe, corner_radius=4)
		frame_map_ctrl.grid_columnconfigure(0, weight=0)
		#frame_map_ctrl.grid_columnconfigure(1, weight=0, minsize=200)
		frame_map_ctrl.grid_columnconfigure(1, weight=1, minsize=200)

		customtkinter.CTkLabel(master=frame_map_ctrl,text="Navigation").grid(row=0, columnspan=3)


		frm = customtkinter.CTkFrame(master=frame_map_ctrl, corner_radius=4)
		frm.grid(row=1,columnspan=2, sticky="n")


		cv = self.canvas

		customtkinter.CTkButton(master=frm, text="↑", command=lambda: cv.move([0,-1]), width=48).grid(row=0, column=1, padx=4,pady=4)
		customtkinter.CTkButton(master=frm, text="←", command=lambda: cv.move([-1,0]), width=48).grid(row=1, column=0, padx=4,pady=4)
		customtkinter.CTkButton(master=frm, text="→", command=lambda: cv.move([1, 0]), width=48).grid(row=1, column=2, padx=4,pady=4)
		customtkinter.CTkButton(master=frm, text="↓", command=lambda: cv.move([0, 1]), width=48).grid(row=2, column=1, padx=4,pady=4)

		customtkinter.CTkButton(master=frm, text="+", width=32, command=lambda: cv.zoom(inc=True) ).grid(row=2,column=0, padx=4,pady=4)
		customtkinter.CTkButton(master=frm, text="-", width=32, command=lambda: cv.zoom(inc=False)).grid(row=2,column=2, padx=4,pady=4)
		

		customtkinter.CTkLabel(master=frame_map_ctrl,text="resolution:", text_font=("Terminal",9)).grid(row=4, column=0, sticky="w")
		customtkinter.CTkLabel(master=frame_map_ctrl, textvariable=self.canvas.variables['resolution']['object'], text_font=("Terminal",9)).grid(row=4, column=1, sticky="e")

		customtkinter.CTkLabel(master=frame_map_ctrl,text="mouse coords:", text_font=("Terminal",9)).grid(row=5, column=0, sticky="w")
		customtkinter.CTkLabel(master=frame_map_ctrl, textvariable=self.canvas.variables['mousepos']['object'], text_font=("Terminal",9)).grid(row=5, column=1, sticky="e")

		customtkinter.CTkLabel(master=frame_map_ctrl,text="scanner coords:", text_font=("Terminal",9)).grid(row=6, column=0, sticky="w")
		customtkinter.CTkLabel(master=frame_map_ctrl, textvariable=self.variables['tippos']['object'], text_font=("Terminal",9)).grid(row=6, column=1, sticky="e")

    
		return frame_map_ctrl

	###########################################################


	def button_function(self):

		print("button pressed")

	def pxsize_change(self,value):

		self.tvar_pxsize.set("{}".format(int(numpy.power(2,value))))

	def phsize_change(self,value):

		vals = [e.value for e in PhysicalSizes]
		s = vals[int(value)]
		u = "nm"

		if s >= 1000:
			s /= 1000
			u = "μm"
		self.tvar_phsize.set("{} {}".format(s,u))

	def phang_change(self,value):

		self.tvar_phang.set("{}°".format(value))


	def scan_click(self):

		print("scanning...")


		# calls Scanner.ScanImage(pixels, size, angle)
		vals = [e.value for e in PhysicalSizes]
		size = vals[int(self.sld_ph.get())]
		npx = int(numpy.power(2, self.sld_px.get()))
		angle = self.sld_angle.get()

		scan = self.ScanFunction(npx, size, angle)
		self.canvas.AddObject(CanvasSPM("spm", scan), noRender=True)
		print("scan completed")

		#plt.matshow(scan.data)
		#plt.show()

		# update the tip position
		tip = self.GetTipFunction()
		self._onTipPosChange(tip)



	def MoveTip(self, event):
		
		p = self.canvas.canvas_to_physical(numpy.asarray([event.x, event.y]))
		#print("canvas click at", event.x, event.y, "--",p)

		self.MoveTipFunction(p)

		# when the movement is done, show the position
		self._onTipPosChange(p)



	def _onTipPosChange(self, newpos):

		p = newpos
		self.variables["tippos"]["value"][0] = p[0]
		self.variables["tippos"]["value"][1] = p[1]
		
		m0, u0, n0 = self.canvas.physical_to_approximate(p[0], 3)
		m1, u1, n1 = self.canvas.physical_to_approximate(p[1], 3)
		self.variables['tippos']['object'].set(
			"x:{:+.3f} {}, y:{:+.3f} {}".format(m0,u0, m1,u1)
		)

		self.canvas.render()


	# DEPRECATED... they are now in the physical canvas object
	'''
	def canvas_redraw(self):

		self.canvas.delete("all")

		self._canvas_compute_corners()


		# draw the spms

		# get the global min/max
		imgmin = float("inf")
		imgmax = float("-inf")
		for spm in self._scans:
			m = numpy.min(spm.data)
			imgmin = min(m, imgmin)

			m = numpy.max(spm.data)
			imgmax = max(m, imgmax)

		self._imgmin = imgmin
		self._imgmax = imgmax

		#print("rendering images...")
		# sort images by resolution
		# low res images are drawn first
		scans = sorted(self._scans, key=lambda x: x.pixelSize[0], reverse=True)
		self._crops = []
		for spm in scans:
			#print("rendering spm...")
			self.canvas_redraw_spm(spm)



		# draw some debug points
		q = numpy.zeros(2)
		p = self.physical_to_canvas(q)
		#print(p,q, self.canvas_size)
		self.canvas.create_oval(p[0]-2, p[1]-2, p[0]+2, p[1]+2, fill="#FF0000")
		
		q[0] = 1
		p = self.physical_to_canvas(q)
		#print(p,q)
		self.canvas.create_oval(p[0]-2, p[1]-2, p[0]+2, p[1]+2, fill="#00FF00")

		q[1] = 1
		p = self.physical_to_canvas(q)
		#print(p,q)
		self.canvas.create_oval(p[0]-2, p[1]-2, p[0]+2, p[1]+2, fill="#0000FF")
		



		# draw the tip guessed position - crosshair
		self.canvas_redraw_tip()


		self.canvas_redraw_scalebar()


	def canvas_redraw_spm(self, spm):

		# get the vertexes of this spm in canvas px coordinates
		# ASSUMPTIONS (DEBUG!!!):
		# 	angle is 0
		# 	slow scan is from bottom to top (positive y axis in physical space)
		# 	fast scan is in the positive x physical axis


		# determine which part of the picture to draw
		# these are the canvas corners in physical space
		y0 = self.canvas_corners[3,1]
		ym = self.canvas_corners[0,1]
		x0 = self.canvas_corners[0,0]
		xm = self.canvas_corners[1,0]

		#print("canvas corners:",[x0,y0],[xm,ym])

		# TODO: include SPM image rotation


		# if both corners of an edge are on the same side of the canvas, the image is out
		spm_x0 = spm.frame_corners[0,0]
		spm_xm = spm.frame_corners[1,0]

		frame_x0 = numpy.max([x0,spm_x0])
		frame_xm = numpy.min([xm,spm_xm])

		#print("frame x",frame_x0,frame_xm,x0,xm)

		if frame_xm < x0 or frame_x0 > xm:
			#print("spm is out of canvas (x)")
			return None

		spm_y0 = spm.frame_corners[0,1]
		spm_ym = spm.frame_corners[2,1]

		frame_y0 = numpy.max([y0,spm_y0])
		frame_ym = numpy.min([ym,spm_ym])

		if frame_ym < y0 or frame_y0 > ym:
			#print("spm is out of canvas (y)")
			return None

		# code here => there is some overlap between spm and canvas
		#print("frame boundaries on spm (x):",[spm_x0,frame_x0],[spm_xm,frame_xm])
		#print("frame boundaries on spm (y):",[spm_y0,frame_y0],[spm_ym,frame_ym])



		# convert height values to color
		# this can make the topography contrast go away quite a bit
		data = spm.data - self._imgmin # also applies the shift
		data /= self._imgmax
		data *= 255

		# final conversion to bytes and flip vertically
		data = data.astype(numpy.uint8)
		data = numpy.flip(data, axis=0)

		# create the PIL image object from data
		pic = Image.fromarray(data)
		rot = pic.rotate(spm.angle, expand=True)
		# make a rotation mask
		mask = numpy.zeros(data.shape,dtype=numpy.uint8)
		mask += 255
		mask = Image.fromarray(mask)
		mask = mask.rotate(spm.angle, expand=True)

		# this is completely white-transparent image to blend with rot using mask
		bgim = numpy.zeros((data.shape[0],data.shape[1],4),dtype=numpy.uint8)
		bgim[:,:,0] = bgim[:,:,1] = bgim[:,:,2] = 255
		bgim = Image.fromarray(bgim, mode="RGBA")
		bgim = bgim.rotate(spm.angle, expand=True)

		rotm = Image.composite(rot, bgim, mask)

		# crop the image
		# where is frame_x0 in spm pixel coordinates?
		frame_px_x0 = int(numpy.floor((frame_x0 - spm_x0) / spm.pixelSize[0]))
		frame_px_xm = int((frame_xm-spm_x0) / spm.pixelSize[0])
		if frame_px_xm == 0: frame_px_xm = 1

		#print("frame pixel coords (x)",frame_px_x0,frame_px_xm)
		#data = data[:,frame_px_x0:frame_px_xm+1]

		frame_px_y0 = int(numpy.floor((frame_y0 - spm_y0) / spm.pixelSize[1]))
		frame_px_ym = int((frame_ym-spm_y0) / spm.pixelSize[1])
		if frame_px_ym == 0: frame_px_ym = 1

		#print("frame pixel coords (y)",frame_px_y0,frame_px_ym)
		#data = data[frame_px_y0:frame_px_ym+1]

		#crop_px = numpy.asarray([data.shape[1],data.shape[0]])
		#crop_nm = crop_px * spm.pixelSize
		#print("cropped size {}px - {}nm".format(crop_px, crop_nm))

		#print("data stats",numpy.mean(spm.data),numpy.min(spm.data),numpy.max(spm.data))

		# perform the crop
		cropbox = (frame_px_x0, rotm.size[1]-frame_px_ym, frame_px_xm, rotm.size[1]-frame_px_y0)
		#print("cropping",rotm.size, cropbox)
		pic = rotm.crop(cropbox)
		

		# resample to match canvas resolution

		# we have to make the spm pixels the same size as the canvas pixels
		# canvas pixel size is 1 / self.canvas_res
		# spm pixel size is spm.pixelSize (x,y components)

		trgPXsize = numpy.asarray([1,1]) / self.canvas_res
		curPXsize = spm.pixelSize
		scaling = curPXsize / trgPXsize
		newsize = numpy.round(numpy.asarray([pic.size[0],pic.size[1]]) * scaling)
		newsize = newsize.astype(numpy.uint32)
		method = Image.Resampling.BICUBIC
		if scaling[0] < 1 and scaling[1] < 1:
			method = Image.Resampling.LANCZOS
		#print(trgPXsize,curPXsize,scaling,"--",data.shape, newsize)

		pic = pic.resize(newsize, resample=method)
		tkpic = ImageTk.PhotoImage(image=pic)

	

		self._crops.append(tkpic)



		p = numpy.asarray([frame_x0, frame_y0], dtype=numpy.float64)
		c = self.physical_to_canvas(p)
		#print("canvas placement:",p,c)
		self.canvas.create_image(c[0],c[1], image=tkpic, anchor="sw")
		#print("the spm is now {}w x {}h [nm]".format(pic.size[0]/self.canvas_res, pic.size[1]/self.canvas_res))
		
		return


	# makes the crosshair at the scanner position
	def canvas_redraw_tip(self):

		tip = self.GetTipFunction()
		ctip = self.physical_to_canvas(tip)
		u = ["nm","nm"]
		for i in range(2):
			if numpy.abs(tip[i]) > 1000:
				tip[i] /= 1000
				u[i] = "μm"

		self.tvar_canvas_scanner.set("x:{:+.3f} {}, y:{:+.3f} {}".format(tip[0],u[0],tip[1],u[1]))
		
		self.canvas.create_line(ctip[0], ctip[1]-8, ctip[0], ctip[1]-2, fill="red")
		self.canvas.create_line(ctip[0], ctip[1]+8, ctip[0], ctip[1]+2, fill="red")

		self.canvas.create_line(ctip[0]-8, ctip[1], ctip[0]-2, ctip[1], fill="red")
		self.canvas.create_line(ctip[0]+8, ctip[1], ctip[0]+2, ctip[1], fill="red")

	'''


class TabLithoPath(customtkinter.CTkFrame):

	def __init__(self, parent, controller):

		customtkinter.CTkFrame.__init__(self, parent)

		self.alanngui = controller
		self.frame_options_dict={} # when we load a GDS file, each shape will get its own frame that will
		# contain options to choose from on how to write. This dictionary will contain those frames
		self.gds = None

		self.grid_rowconfigure(0, weight=1)
		self.grid_columnconfigure(0, weight=0, minsize=400)
		self.grid_columnconfigure(1, weight=2, minsize=400)


		### panel variables
		self.variables = {
			'writefield': 	{'object': tk.StringVar(self), 'value':None},
			'writespeed': 	{'object': tk.StringVar(self), 'value':None},
			'idlespeed': 	{'object': tk.StringVar(self), 'value':None},
			'pitch': 		{'object': tk.StringVar(self), 'value':None},
			'write_gap_voltage': {'object': tk.StringVar(self), 'value':None},
			'write_current_setpoint': {'object': tk.StringVar(self), 'value':None},
			'path_dwell_time': {'object': tk.StringVar(self), 'value':None}
		}
		self.polygons = []
		self.lith_paths = [] # write paths inside the polygons
		self.inbetween_paths = [] # write paths between polygons (in order)
		self.crosshairs = [] #crosshairs for inbetween paths for checking

		###########################
		# CANVAS FRAME
		###########################
		canvas = PhysicalCanvas(self, background="white")
		canvas.grid(row=0, column=1,padx=8,pady=8, sticky="nsew")
		self.canvas = canvas

		canvas.callbacks['click'].append(self._canvas_onclick)

		#canvas.AddObject(CanvasLine("",numpy.asarray([[0,0],[5,5]]), fill="red"))
		#canvas.AddObject(CanvasPoint("",numpy.asarray([0,0]), pxsize=2, fill="blue"))




		#######################################
		# frame for path controls
		#######################################
		panel = customtkinter.CTkFrame(self)
		panel.grid(row=0,column=0, padx=8, pady=8, sticky='nsew')
		panel.grid_columnconfigure(0, weight=1)
		self.rast_prop = panel
		self.mainpanel = panel

		### title
		customtkinter.CTkLabel(panel, text="LithoPath Controls", text_font=('Roboto', 14)).grid(row=0, column=0, pady=4, padx=4, sticky='new')

		### load file button
		self.gdsLoaded = False
		customtkinter.CTkButton(panel, text='Load file', command=self.openfile_onclick).grid(row=1,column=0,sticky="n")


		### control panel
		self.controlpanel = self._init_controlPanel(panel)
		self.controlpanel.grid(row=2, padx=4,pady=4,sticky="new")
		
		### shape control panel
		self.shapepanel = self._init_shape_panel(panel)
		self.shapepanel.grid(row=3, padx=4, pady=4, sticky="new")
		
		### navigation panel
		self.navpanel = self._init_nav_panel(panel)
		self.navpanel.grid(row=4,column=0, padx=4, pady=4, sticky="new")



	### SUB PANELS ### ########################################

	def _init_controlPanel(self, master):

		cp = customtkinter.CTkFrame(master)
		cp.grid_columnconfigure(0, weight=1)
		cp.grid_columnconfigure(1, weight=1)

		customtkinter.CTkLabel(cp, text="Write settings").grid(row=0, columnspan=2, pady=4, sticky='new')


		# not needed for now
		#customtkinter.CTkLabel(cp, text="Write Field Size [nm]: ").grid(row=1, column=0, pady=4, sticky='w')
		#self.control_writefield = customtkinter.CTkEntry(cp, textvariable=self.variables['writefield']['object'])
		#self.control_writefield.grid(row=1, column=1, padx=4, sticky='ew')
		
		customtkinter.CTkLabel(cp, text="Pitch [nm]: ").grid(row=2, column=0, pady=4, sticky='w')
		self.control_pitch = customtkinter.CTkEntry(cp, textvariable=self.variables['pitch']['object'])
		self.control_pitch.grid(row=2, column=1, padx=4, sticky='ew')

		customtkinter.CTkLabel(cp, text="Write Speed [nm/s]: ").grid(row=3, column=0, pady=4, sticky='w')		
		self.control_writespeed = customtkinter.CTkEntry(cp, textvariable=self.variables['writespeed']['object'])
		self.control_writespeed.grid(row=3, column=1, padx=4, sticky='ew')
		
		customtkinter.CTkLabel(cp, text="Idle Speed [nm/s]: ").grid(row=4, column=0, pady=4,sticky='w')
		self.control_idlespeed = customtkinter.CTkEntry(cp, textvariable=self.variables['idlespeed']['object'])
		self.control_idlespeed.grid(row=4, column=1, padx=4, sticky='ew')

		customtkinter.CTkLabel(cp, text="Path Dwell Time [s]: ").grid(row=5, column=0, pady=4,sticky='w')
		self.control_idlespeed = customtkinter.CTkEntry(cp, textvariable=self.variables['path_dwell_time']['object'])
		self.control_idlespeed.grid(row=5, column=1, padx=4, sticky='ew')

		customtkinter.CTkLabel(cp, text="Write Gap Voltage [V]: ").grid(row=6, column=0, pady=4,sticky='w')
		self.control_idlespeed = customtkinter.CTkEntry(cp, textvariable=self.variables['write_gap_voltage']['object'])
		self.control_idlespeed.grid(row=6, column=1, padx=4, sticky='ew')

		customtkinter.CTkLabel(cp, text="Write Current Setpoint [A]: ").grid(row=7, column=0, pady=4,sticky='w')
		self.control_idlespeed = customtkinter.CTkEntry(cp, textvariable=self.variables['write_current_setpoint']['object'])
		self.control_idlespeed.grid(row=7, column=1, padx=4, sticky='ew')

		#customtkinter.CTkCheckBox(cp, text="Invert image").grid(row=5, columnspan=2,pady=8, sticky='n')

		customtkinter.CTkButton(cp, text='Export', command=self.export_onclick).grid(row=10, column=0, pady=4, sticky="n")
		customtkinter.CTkButton(cp, text='Combine Paths', command=self.combine_paths).grid(row=10, column=1, pady=4, sticky="n")
		

		return cp

	def _init_nav_panel(self, mainframe):

		frame_map_ctrl = customtkinter.CTkFrame(master=mainframe, corner_radius=4)
		frame_map_ctrl.grid_columnconfigure(0, weight=0)
		frame_map_ctrl.grid_columnconfigure(1, weight=1, minsize=200)

		customtkinter.CTkLabel(master=frame_map_ctrl,text="Navigation").grid(row=0, columnspan=2, sticky="n")

		frm = customtkinter.CTkFrame(master=frame_map_ctrl, corner_radius=4)
		frm.grid(row=1,columnspan=2, sticky="n")


		cv = self.canvas

		customtkinter.CTkButton(master=frm, text="↑", command=lambda: cv.move([0,-1]), width=48).grid(row=0, column=1, padx=4,pady=4)
		customtkinter.CTkButton(master=frm, text="←", command=lambda: cv.move([-1,0]), width=48).grid(row=1, column=0, padx=4,pady=4)
		customtkinter.CTkButton(master=frm, text="→", command=lambda: cv.move([1, 0]), width=48).grid(row=1, column=2, padx=4,pady=4)
		customtkinter.CTkButton(master=frm, text="↓", command=lambda: cv.move([0, 1]), width=48).grid(row=2, column=1, padx=4,pady=4)

		customtkinter.CTkButton(master=frm, text="+", width=32, command=lambda: cv.zoom(inc=True) ).grid(row=2,column=0, padx=4,pady=4)
		customtkinter.CTkButton(master=frm, text="-", width=32, command=lambda: cv.zoom(inc=False)).grid(row=2,column=2, padx=4,pady=4)
			

		customtkinter.CTkLabel(master=frame_map_ctrl,text="resolution:", text_font=("Terminal",9)).grid(row=2, column=0, sticky="w")
		customtkinter.CTkLabel(master=frame_map_ctrl, textvariable=self.canvas.variables['resolution']['object'], text_font=("Terminal",9)).grid(row=2, column=1, sticky="e")

		customtkinter.CTkLabel(master=frame_map_ctrl,text="mouse coords:", text_font=("Terminal",9)).grid(row=3, column=0, sticky="w")
		customtkinter.CTkLabel(master=frame_map_ctrl, textvariable=self.canvas.variables['mousepos']['object'], text_font=("Terminal",9)).grid(row=3, column=1, sticky="e")

		
		return frame_map_ctrl

	def _init_shape_panel(self, master):

		frm = customtkinter.CTkFrame(master=master, corner_radius=4)
		frm.grid_columnconfigure(0, weight=0)
		frm.grid_columnconfigure(1, weight=1)

		self.variables['shape_selected'] = {'object': tk.StringVar(frm), 'value': None}
		self.variables['shape_filltype'] = {'object': tk.StringVar(frm), 'value': None}
		self.variables['shape_outline'] = {'object': tk.StringVar(frm), 'value': None}
		self.variables['shape_Write'] = {'object': None, 'value': None}
		self.variables['shape_clear'] = {'object': None, 'value': None}


		self.variables['shape_selected']['object'].set('Shape Write')
		label = customtkinter.CTkLabel(frm, textvariable=self.variables['shape_selected']['object'], text_font=('Helvetica', 10)).grid(row=0, columnspan=2, pady=4, padx=4, sticky='n')
		
		
		options_scan = ['X-serpentine', 'Y-serpentine', 'Spiral', 'more tbc']
		customtkinter.CTkLabel(frm, text="Fill type: ", text_font=('Helvetica', 10)).grid(row=1, column=0, pady=4, padx=4,sticky='w')
		self.variables['shape_filltype']['value'] = ttk.OptionMenu(frm, self.variables['shape_filltype']['object'], options_scan[0], *options_scan )
		self.variables['shape_filltype']['value'].grid(row=1, column=1, pady=4,padx=4, sticky='e')
		#self.variables['shape_filltype']['value'].config(state=tk.DISABLED)
		

		options_fill = ['Only fill', 'Fill and outline']
		customtkinter.CTkLabel(frm, text="Mode: ", text_font=('Helvetica', 10)).grid(row=2, column=0, pady=4, padx=4, sticky='w')
		self.variables['shape_outline']['value'] = ttk.OptionMenu(frm, self.variables['shape_outline']['object'], options_fill[0], *options_fill)
		self.variables['shape_outline']['value'].grid(row=2, column=1, pady=4,padx=4, sticky='e')
		#self.variables['shape_outline']['value'].config(state=tk.DISABLED)


		self.variables['shape_Write']['value'] = customtkinter.CTkButton(master=frm, text="Write", command=self._Write_onclick)
		self.variables['shape_Write']['value'].grid(row=3, column=0, padx=4,pady=4, sticky="ne")
		#self.variables['shape_Write']['value'].config(state=tk.DISABLED)

		Apply_to_all = customtkinter.CTkButton(master=frm, text="Apply to all", command=self._Apply_all)
		Apply_to_all.grid(row=3, column=1, padx=4,pady=4, sticky="ne")

		self.variables['shape_clear']['value'] = customtkinter.CTkButton(master=frm, text="Reset", command=self._clear_onclick)
		self.variables['shape_clear']['value'].grid(row=3, column=2, padx=4,pady=4, sticky="nw")
		#self.variables['shape_clear']['value'].config(state=tk.DISABLED)

		

		
		return frm

	###########################################################


	## called when the load button is clicked
	def openfile_onclick(self):


		file = filedialog.askopenfile(mode='r')
		print("selected file", file)

		if file is None:
			return


		# code here => a file was selected
		self.gds = GDSConverter.GDS(file)
		file.close()


		# hopefully the file was opened and parsed correctly!
		# show the polygons on the canvas
		self.polygons = []
		mean = numpy.zeros(2, dtype=numpy.float64)
		minmax = numpy.zeros((2,2), dtype=numpy.float64)
		minmax[0,:] = float("inf")
		minmax[1,:] = float("-inf")
		
		for shapeID in self.gds.shapes.keys():

			shape = self.gds.shapes[shapeID]

			# polygon of the starting shape
			poly = CanvasLine("poly[{}]".format(shapeID), shape.vertexes, fill="blue")
			poly.srcShape = shape

			self.polygons.append(poly)
			self.canvas.AddObject(poly, noRender=True)

			m = numpy.mean(shape.vertexes[0:-1], axis=0) # avoid the last point since it is same as first
			mean += m

			m = numpy.min(numpy.concatenate((shape.vertexes, [minmax[0]]), axis=0), axis=0)
			minmax[0] = m
			m = numpy.max(numpy.concatenate((shape.vertexes, [minmax[1]]), axis=0), axis=0)
			minmax[1] = m
			
			# make the frames containing the option menus for all the shapes but do not render them on screen
			# they will be rendered only when the mouse clicks within one of the shapes
			#self.make_shape_frame(shapeID)


			# TODO: add Writeization if done - NO! the file was just loaded so there cannot be a Writeization available


		# rescale and recenter the
		mean /= len(self.gds.shapes)
		self.canvas.center = mean

		# we want canvas.physicalside to be bigger than max-min
		desiredSide = numpy.max(minmax[1]-minmax[0])*1.1
		currentSidePx = numpy.min(self.canvas.size)
		currentSide = currentSidePx / self.canvas.resolution
		desiredSideRes = currentSidePx / desiredSide
		#print("res: ",currentSide,currentSidePx,desiredSide,desiredSideRes)

		# this will also redraw the canvas
		self.canvas.setSpace(mean, desiredSideRes)



	def _canvas_onclick(self, event):

		# no gds = no fun
		if not self.gds:
			return

		c = numpy.asarray([event.x, event.y])
		p = self.canvas.canvas_to_physical(c)
		#print("check polygons at",c,p)

		selected = None

		for shapeID in self.gds.shapes.keys():
			shape = self.gds.shapes[shapeID]
			#print("checking poly",shapeID)

			a = [x for x in self.polygons if x.srcShape == shape]
			poly = a[0]

			if shape.pointIsInside(p):

				#print("selected polygon",shapeID)
				selected = shapeID

				# mark as selected
				poly.options['fill'] = 'red'
				poly.options['width'] = 3

				#self.frame_options_dict[shapeID].grid(row=10, columnspan=2, sticky='n')
				self.variables['shape_selected']['object'].set("Shape[{}] Writeization".format(shapeID))
				self.variables['shape_selected']['value'] = shape

				# load the Writeization options if available
				self.variables['shape_filltype']['value'].config(state=tk.NORMAL)
				self.variables['shape_outline']['value'].config(state=tk.NORMAL)
				self.variables['shape_Write']['value'].config(state=tk.NORMAL)
				self.variables['shape_clear']['value'].config(state=tk.NORMAL)
				# ...
				if shape.writePath is not None:
					self.variables['shape_filltype']['object'].set(shape.writeType)
					self.variables['shape_outline']['object'].set(shape.writeMode)
					
				
			else:

				# deselect the polygon
				poly.options['fill'] = 'blue'
				poly.options['width'] = 1

		
		
		if selected is None:

			self.variables['shape_selected']['object'].set("Shape[not selected] Writeization")
			self.variables['shape_selected']['value'] = None

			# disable controls
			#self.variables['shape_filltype']['value'].config(state=tk.DISABLED)
			#self.variables['shape_outline']['value'].config(state=tk.DISABLED)
			#self.variables['shape_Write']['value'].config(state=tk.DISABLED)
			#self.variables['shape_clear']['value'].config(state=tk.DISABLED)
			# need these enabled so that we can use the apply to all button
		
		self.canvas.render()

		
		


	def exptype_onchange(self, variable):

		"""
		Export type selector onchange event handler.
		This is called automatically by the GUI.

		:param variable: selected format option
		:type variable: str
		
		"""


		if variable=='.txt file':
			self.control_writefield.config(state=tk.DISABLED)
			self.control_writespeed.config(state=tk.DISABLED)
			self.control_idlespeed.config(state=tk.DISABLED)
		
		elif variable=='Matrix Script':
			self.control_writefield.config(state=tk.NORMAL)
			self.control_writespeed.config(state=tk.NORMAL)
			self.control_idlespeed.config(state=tk.NORMAL)

		else:
			raise ValueError("Export type not implemented")


	def _Write_onclick(self):

		shape = self.variables['shape_selected']['value']
		if shape is None: return

		# remove any inbetween (idle) paths
		for path in self.inbetween_paths: 
			self.canvas.RemoveObject(path.name)

		# remove old Write from view - do not update
		self.canvas.RemoveObject('polyfill-{}'.format(shape.index), noRender=True)
		# remove old crosshairs
		self.canvas.RemoveObject('crosshair-{}0'.format(shape.index), noRender=True)
		self.canvas.RemoveObject('crosshair-{}1'.format(shape.index), noRender=True)


		# compute the Write
		rtype = self.variables['shape_filltype']['object'].get()
		routl = self.variables['shape_outline']['object'].get()
		pitch = int(self.variables['pitch']['object'].get())
		shape.writeType = rtype
		shape.writeMode = routl

		shape.vector_scan(rtype, routl, pitch)
		

		# add new Write to the canvas
		fillLine = CanvasLine("polyfill-{}".format(shape.index), shape.writePath, fill="orange")
		self.canvas.AddObject(fillLine)
		# add starting and ending crosshairs so can tell if it matches with the inbetween lines (just for testing atm)
		crosshair0 = CanvasCrossHair("crosshair-{}0".format(shape.index), shape.writePath[0,:], fill='red') #start
		crosshair1 = CanvasCrossHair("crosshair-{}1".format(shape.index), shape.writePath[-1,:], fill='blue') #end
		self.canvas.AddObject(crosshair0)
		self.canvas.AddObject(crosshair1)
		self.crosshairs.append(crosshair0)
		self.crosshairs.append(crosshair1)
		

	def _Apply_all(self):
		# applies selected options too all shapes

		#clear old inbetween paths + their cross hairs
		for path in self.inbetween_paths: 
			self.canvas.RemoveObject(path.name)
		for crosshair in self.crosshairs:
			self.canvas.RemoveObject(crosshair.name)

		for shape in self.gds.shapes:
			rtype = self.variables['shape_filltype']['object'].get()
			routl = self.variables['shape_outline']['object'].get()
			pitch = int(self.variables['pitch']['object'].get())
			self.gds.shapes[shape].writeType = rtype
			self.gds.shapes[shape].writeMode = routl

			self.gds.shapes[shape].vector_scan(rtype, routl, pitch)
			
			# remove old path and crosshairs
			self.canvas.RemoveObject("polyfill-{}".format(shape))
			self.canvas.RemoveObject("crosshair-{}0".format(shape))
			self.canvas.RemoveObject("crosshair-{}1".format(shape))

			# add new Write to the canvas
			fillLine = CanvasLine("polyfill-{}".format(shape), self.gds.shapes[shape].writePath, fill="orange")
			self.canvas.AddObject(fillLine)
			# add starting and ending crosshairs so can tell if it matches with the inbetween lines (just for testing atm)
			crosshair0 = CanvasCrossHair("crosshair-{}0".format(shape), self.gds.shapes[shape].writePath[0,:], fill='red') #start
			crosshair1 = CanvasCrossHair("crosshair-{}1".format(shape), self.gds.shapes[shape].writePath[-1,:], fill='blue') #end
			self.canvas.AddObject(crosshair0)
			self.canvas.AddObject(crosshair1)



	def _clear_onclick(self):

		shape = self.variables['shape_selected']['value']
		if shape is None: return

		# clear writing parameters
		shape.writePath = None
		shape.writeType = None
		shape.writeMode = None

		# remove Writeization from canvas
		self.canvas.RemoveObject('polyfill-{}'.format(shape.index))



	def export_onclick(self):

		# exports the gds to the main navigation page 
		# for now, it will export the vector path to a MATE or .txt file
		mate_file = open('Pattern_Writer_v2_1.mate')
		string_list = mate_file.readlines()
		mate_file.close

		string_list[47] = "var path_dwell_time         = {};           // Dwell Time between writing each path [s] \n".format(self.variables['path_dwell_time']['object'].get())
		string_list[48] = "var write_gap_voltage       = {};         // Writing Voltage [Volts] FEM 6-8 V, APM 3-4 V \n".format(self.variables['write_gap_voltage']['object'].get())
		string_list[49] = "var write_current_setpoint  = {};   // Writing Current [Ampere] FEM 0.75-1.5 nA, APM 3-4.5 nA \n".format(self.variables['write_current_setpoint']['object'].get())
		string_list[50] = "var write_scanspeed         = {};          // Writing Scan Speed [nm/s] 25 - 400 nm/s \n".format(self.variables['writespeed']['object'].get())
		string_list[51] = "var idle_scanspeed          = {};         // Idle Scan Speed whilst not writing [nm/s] \n".format(self.variables['idlespeed']['object'].get())
		
		string_list[260] = 'path=' + str(self.final_list_points) +'\n'
		string_list[261] = 'points=' + str(self.final_list_numpoints) +'\n'

		mate_file = open('Pattern_Writer_v2_1.mate','w')
		new_file_contents = "".join(string_list)
		mate_file.write(new_file_contents)
		mate_file.close()


	
	def combine_paths(self):
		# once all shapes have had their write paths determined, this takes them all in and 
		# decide which order to write the shapes in. It determines this by looking at every 
		# possible path and choosing the one with the smallest average distances.
		points_list = []
		for shape in self.gds.shapes:
			points_list.append( self.gds.shapes[shape].writePath[0,:] )
			points_list.append( self.gds.shapes[shape].writePath[-1,:] )

		points_list = numpy.asarray(points_list)
		dist_dict={}
		n = points_list.shape[0]
		for i in range(n):
			point = points_list[i,:]
			#creates a dictionary that has the distances from each point to every other point
			#because the points come in pairs, there should actually be more zeros than this has worked out 
			#i.e. if the tip goes to the zeroth point, it then moves to the next point while being 'on' so
			#the algorithm should count this as 0. For points with keys, i, that are even this should be the ith distance.
			#for points with keys, j, that are off, this shoulds be the (j-1)th distance. this will be taken
			# care of in the function that calculates all the paths' distances.
			dist_dict[i] = numpy.sqrt(numpy.sum((points_list-point)**2, axis=1)) #distances between all points
		distances = []
		all_perms=self.list_perms(n)
		for path in all_perms:
			dist=[]
			prev_ind = path[0][1]
			for l in path[1:]:
				cur_ind = l[0]
				dist.append(dist_dict[prev_ind][cur_ind])
				prev_ind = l[1]
			distances.append(dist)
		best_ind = numpy.argmin(numpy.mean(distances, axis=1))
		best_perm = [x for sub in all_perms[best_ind] for x in sub]

		# this is the best order for paths between shapes
		best_order =  [list(points_list[i]) for i in best_perm]
		# how to get shape order, not just points order...?
		# find shape order
		best_shape_order = [(y[0]//2 +1) for y in all_perms[best_ind]]

		# remove any old plotted inbetween paths and cross hairs
		for path in self.inbetween_paths: 
			self.canvas.RemoveObject(path.name, noRender=True)
		for crosshair in self.crosshairs:
			self.canvas.RemoveObject(crosshair)

		# now that we've found the shortest order, we need to make sure the start and end points of shape fill in
		# agree with this order. 
		for shape, i in zip(best_shape_order, range(n//2)):
			if numpy.all(self.gds.shapes[shape].writePath[-1,:] == best_order[2*i]):
				self.gds.shapes[shape].writePath = self.gds.shapes[shape].writePath[::-1,:]
				
		
			self.canvas.RemoveObject("crosshair-{}0".format(shape))
			self.canvas.RemoveObject("crosshair-{}1".format(shape))
			crosshair0 = CanvasCrossHair("crosshair-{}0".format(shape), self.gds.shapes[shape].writePath[0,:], fill='red') #start
			crosshair1 = CanvasCrossHair("crosshair-{}1".format(shape), self.gds.shapes[shape].writePath[-1,:], fill='blue') #end
			self.canvas.AddObject(crosshair0)
			self.canvas.AddObject(crosshair1)


		# add cross hairs now too for checking order is correct. Green for idle paths
		for j in range((n//2)-1):
			i=2*j+1
			inbetween_path = CanvasLine("inbetween path" + str(i),numpy.asarray([best_order[i], best_order[i+1] ]), fill="green")
			self.canvas.AddObject( inbetween_path )
			self.inbetween_paths.append( inbetween_path )
			crosshair = CanvasCrossHair("crosshair-{}".format(i), best_order[i], fill='green') #start
			self.canvas.AddObject(crosshair)
			self.crosshairs.append(crosshair)

		# combine inbetween (idle) paths with the write paths into a list of arrays in the right order
		# this is what will be plotted and used in the final producst
		self.final_path = [self.gds.shapes[i].writePath for i in self.gds.shapes]

		# for now, we make another version that gets inserted into to Procopi's MATE script
		# Some info from the MATE script below on formatting of the points
		'''
		STRUCTURE OF THE PATTERN MATRIX in MATE
		path[][][] is a three dimensional array where each row contains a path that shall be traced by the tip in writing mode. Each path consists an array of coorinates (x and y). The number of points per path has to be stored in points[]. 
		The coordinate system origin has to be the top left and the coordinates shall be from 0 to 1 in x and y direction (relative coordinates to selected area!)
		points[] contains the number of points the corresponding path has.
		nrpaths contains the total of independent paths that are written.
		HOW THE AREA OF WRITING IS DEFINED 
		Three points of a quadrilateral shape are defined, within which the pattern shall be written inside. The points are defined using mouse clicks.
		'''

		# translate all shapes so that origin is to the top left corner
		# find highest and left most points
		xcoords = [self.gds.shapes[i].writePath[:,0] for i in self.gds.shapes]
		ycoords = [self.gds.shapes[i].writePath[:,1] for i in self.gds.shapes]
		x_min = min( [numpy.min(array) for array in xcoords])
		y_max = max( [numpy.max(array) for array in ycoords]) 
		# make a dictionary of shapes to use just in this temporary bit of code. Thought it'd be easier to make a standalone dictionary not connected
		# to anything else so getting rid of it later will be easier and won't need to think if it's connected to any other bits of code
		shapes = {}
		for i in self.gds.shapes:
			shapes[i] = self.gds.shapes[i].writePath - numpy.array([x_min, y_max]) # translate
			shapes[i][:,1] = (-1)*shapes[i][:,1] # invert the y coordinates so they increase from 0 downwards
		# now we need to normalise them so that all coords are between 0 and 1
		x_max = max( [numpy.max(array) for array in xcoords])
		y_min = min( [numpy.min(array) for array in ycoords])
		for i in shapes:
			shapes[i][:,1] = shapes[i][:,1]/(y_max-y_min)
			shapes[i][:,0] = shapes[i][:,0]/(x_max-x_min)
		# combine all the paths into one list, but we have to have the paths as lists of points now because the final format needs commas inbetween points and within the points too
		self.final_list_points = [shapes[i].tolist() for i in shapes]
		# we also need the number of points in each shape path
		self.final_list_numpoints = [shapes[i].shape[0] for i in shapes]
		# we save these to the class so that they can be used in the export button

	def list_perms(self, n):
		# Takes in an integer n (number of shapes). Each shape has 2 points associated 
		# with it (start and end of serpentine). Returns every permutation of n pairs
		# and sub permutations of the points within the pairs.
		# e.g.[[0,1][2,3]] --> [[0,1],[2,3]], [[1,0],[2,3]], [[0,1],[3,2]],...
		# think for final product these should be pre-calculated and in a separate file since anything over n~12 takes way too long to wait for every time
		# for anything over something like n=20, might be best to just take the next shortest distance
		b = numpy.array([[0,1]])
		for i in range(n//2-1):
			b = numpy.vstack((b,b[-1]+2))
		perms = [list(map(list,it.permutations(subl))) for subl in b]
		# we apply it.permutations to every sublist in b (i.e. [1,2], [2,3] etc), and then join these lists
		product_list = [list(map(list, data)) for data in it.product(*perms)]
		# we use it.product to give us the different products of these
		all_perms=[list(map(list, it.permutations(l))) for l in product_list]
		return [x for sub in all_perms for x in sub]







if __name__ == "__main__":

	print("Hello world!")


	'''
	a = [[0,1,2,3],[1,2,3,4],[2,3,4,5],[3,4,5,6]]
	a = numpy.asarray(a, dtype=numpy.float64)
	a /= 6
	a *= 255
	a = a.astype(numpy.uint8)
	a = numpy.flip(a, axis=0)
	pic = Image.fromarray(a)
	pic.save("test.png")
	'''


	# create a sample
	s = Sample(20000, 5.0, 0.5, 1.8, 8)
	#s = SampleCheck(10.0)

	# make a scanner
	scn = Scanner(s)


	# create the gui
	gui = ALANNGUI()
	# assign a scan function
	gui.tabInfo['nav']['frame'].ScanFunction = scn.ScanImage
	gui.tabInfo['nav']['frame'].MoveTipFunction = scn.MoveTip
	gui.tabInfo['nav']['frame'].GetTipFunction = scn.GetTip


	# run the app
	gui.mainloop()
