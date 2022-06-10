import tkinter as tk
from tkinter import ttk
from tkinter import filedialog # needed so we can open a file explorer window when looking for the .gds/.bmp/etc files
import customtkinter

import numpy
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
<<<<<<< Updated upstream
		data /= canvas.SPM_max/1.5
=======
		data /= canvas.SPM_max
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
			'click': [],

=======
			'click': []
>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
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


=======
>>>>>>> Stashed changes
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
		data /= canvas.SPM_max
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
			'click': []
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
<<<<<<< Updated upstream
<<<<<<< Updated upstream

			tab['button'] = customtkinter.CTkButton(menu, text=tab['name'], command=cmd, width=self.menu_width, text_color_disabled="black")
			tab['button'].grid(row=0, column=col, padx=2, pady=2)
			tab['button'].configure(fg_color="#4682bd")

			frame = globals()[tab['class']](container, self)
			frame.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
			tab['frame'] = frame

			col += 1

		self.tab_show('nav')

=======

			tab['button'] = customtkinter.CTkButton(menu, text=tab['name'], command=cmd, width=self.menu_width, text_color_disabled="black")
			tab['button'].grid(row=0, column=col, padx=2, pady=2)
			tab['button'].configure(fg_color="#4682bd")

			frame = globals()[tab['class']](container, self)
			frame.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
			tab['frame'] = frame

			col += 1

		self.tab_show('nav')

>>>>>>> Stashed changes

	def tab_show(self, tabname):

		# brings forward the frame of the tab you want to see
		tab = self.tabInfo[tabname]

		for tn in self.tabInfo.keys():
			if tn != tab['class']:
				self.tabInfo[tn]['button'].configure(fg_color="#4682bd", state=tk.NORMAL)
				if self.tabInfo[tn]['frame'].canvas:
					self.tabInfo[tn]['frame'].canvas.lose_focus()

=======

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

>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes

=======

<<<<<<< Updated upstream

>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
		# and this is the map panel - for the canvas
		frame_map = customtkinter.CTkFrame(master=self, corner_radius=4)
		frame_map.grid(row=0, column=1, padx=4, pady=4, sticky="nsew")

		frame_map.grid_columnconfigure(0, weight=2)
		frame_map.grid_rowconfigure(0, weight=2)

		# canvas - this has to go first
		canvas = PhysicalCanvas(frame_map, background="white")
		canvas.grid(row=0, column=0,padx=4,pady=4, sticky="nsew")
		self.canvas = canvas



<<<<<<< Updated upstream
<<<<<<< Updated upstream
		# this should be the main control bar on the left
		frame_ctrl = customtkinter.CTkFrame(master=self, width=250, corner_radius=4)
		frame_ctrl.grid(row=0, column=0, padx=8, pady=8, sticky="nsew")
		#frame_ctrl.grid_propagate(False)
=======
=======
>>>>>>> Stashed changes

		# this should be the main control bar on the left
		frame_ctrl = customtkinter.CTkFrame(master=self, width=250, corner_radius=4)
		frame_ctrl.grid(row=0, column=0, padx=8, pady=8, sticky="nsew")
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
		self.frame_ctrl = frame_ctrl

		customtkinter.CTkLabel(master=frame_ctrl,text="Navigation & Mapping", text_font = ("Roboto",14)).grid(row=0, column=0, pady=4)

		frm_scan = self._init_scan_panel(frame_ctrl)
<<<<<<< Updated upstream
<<<<<<< Updated upstream
		frm_scan.grid(row=1, column=0, pady=4, padx=4,sticky="new")
=======
		frm_scan.grid(row=1, column=0, pady=4, padx=4)
>>>>>>> Stashed changes
=======
		frm_scan.grid(row=1, column=0, pady=4, padx=4)
>>>>>>> Stashed changes
		



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
<<<<<<< Updated upstream
<<<<<<< Updated upstream
		frame_map_ctrl.grid_columnconfigure(0, weight=0)
		frame_map_ctrl.grid_columnconfigure(1, weight=0, minsize=200)
		frame_map_ctrl.grid_columnconfigure(1, weight=1, minsize=20)

		customtkinter.CTkLabel(master=frame_map_ctrl,text="Navigation").grid(row=0, columnspan=3)
=======
		frame_map_ctrl.grid_columnconfigure(0, weight=1, minsize=200)
		frame_map_ctrl.grid_columnconfigure(1, weight=1, minsize=200)

		customtkinter.CTkLabel(master=frame_map_ctrl,text="Navigation").grid(row=0, columnspan=2)
>>>>>>> Stashed changes

		frm = customtkinter.CTkFrame(master=frame_map_ctrl, corner_radius=4)
		frm.grid(row=1,columnspan=2, sticky="n")

<<<<<<< Updated upstream

		cv = self.canvas

		customtkinter.CTkButton(master=frm, text="↑", command=lambda: cv.move([0,-1]), width=48).grid(row=0, column=1, padx=4,pady=4)
		customtkinter.CTkButton(master=frm, text="←", command=lambda: cv.move([-1,0]), width=48).grid(row=1, column=0, padx=4,pady=4)
		customtkinter.CTkButton(master=frm, text="→", command=lambda: cv.move([1, 0]), width=48).grid(row=1, column=2, padx=4,pady=4)
		customtkinter.CTkButton(master=frm, text="↓", command=lambda: cv.move([0, 1]), width=48).grid(row=2, column=1, padx=4,pady=4)

=======
		frame_map_ctrl.grid_columnconfigure(0, weight=1, minsize=200)
		frame_map_ctrl.grid_columnconfigure(1, weight=1, minsize=200)

		customtkinter.CTkLabel(master=frame_map_ctrl,text="Navigation").grid(row=0, columnspan=2)

		frm = customtkinter.CTkFrame(master=frame_map_ctrl, corner_radius=4)
		frm.grid(row=1,columnspan=2, sticky="n")


		cv = self.canvas

		customtkinter.CTkButton(master=frm, text="↑", command=lambda: cv.move([0,-1]), width=48).grid(row=0, column=1, padx=4,pady=4)
		customtkinter.CTkButton(master=frm, text="←", command=lambda: cv.move([-1,0]), width=48).grid(row=1, column=0, padx=4,pady=4)
		customtkinter.CTkButton(master=frm, text="→", command=lambda: cv.move([1, 0]), width=48).grid(row=1, column=2, padx=4,pady=4)
		customtkinter.CTkButton(master=frm, text="↓", command=lambda: cv.move([0, 1]), width=48).grid(row=2, column=1, padx=4,pady=4)

>>>>>>> Stashed changes
=======

		cv = self.canvas

		customtkinter.CTkButton(master=frm, text="↑", command=lambda: cv.move([0,-1]), width=48).grid(row=0, column=1, padx=4,pady=4)
		customtkinter.CTkButton(master=frm, text="←", command=lambda: cv.move([-1,0]), width=48).grid(row=1, column=0, padx=4,pady=4)
		customtkinter.CTkButton(master=frm, text="→", command=lambda: cv.move([1, 0]), width=48).grid(row=1, column=2, padx=4,pady=4)
		customtkinter.CTkButton(master=frm, text="↓", command=lambda: cv.move([0, 1]), width=48).grid(row=2, column=1, padx=4,pady=4)

>>>>>>> Stashed changes
		customtkinter.CTkButton(master=frm, text="+", width=32, command=lambda: cv.zoom(inc=True) ).grid(row=2,column=0, padx=4,pady=4)
		customtkinter.CTkButton(master=frm, text="-", width=32, command=lambda: cv.zoom(inc=False)).grid(row=2,column=2, padx=4,pady=4)
		

<<<<<<< Updated upstream
		customtkinter.CTkLabel(master=frame_map_ctrl,text="resolution:", text_font=("Terminal",9)).grid(row=4, column=0, sticky="w")
		customtkinter.CTkLabel(master=frame_map_ctrl, textvariable=self.canvas.variables['resolution']['object'], text_font=("Terminal",9)).grid(row=4, column=1, sticky="e")

		customtkinter.CTkLabel(master=frame_map_ctrl,text="mouse coords:", text_font=("Terminal",9)).grid(row=5, column=0, sticky="w")
		customtkinter.CTkLabel(master=frame_map_ctrl, textvariable=self.canvas.variables['mousepos']['object'], text_font=("Terminal",9)).grid(row=5, column=1, sticky="e")

		customtkinter.CTkLabel(master=frame_map_ctrl,text="scanner coords:", text_font=("Terminal",9)).grid(row=6, column=0, sticky="w")
		customtkinter.CTkLabel(master=frame_map_ctrl, textvariable=self.variables['tippos']['object'], text_font=("Terminal",9)).grid(row=6, column=1, sticky="e")
=======
		customtkinter.CTkLabel(master=frame_map_ctrl,text="resolution:", text_font=("Terminal",9)).grid(row=4, column=0)
		customtkinter.CTkLabel(master=frame_map_ctrl, textvariable=self.canvas.variables['resolution']['object'], text_font=("Terminal",9)).grid(row=4, column=1)

		customtkinter.CTkLabel(master=frame_map_ctrl,text="mouse coords:", text_font=("Terminal",9)).grid(row=5, column=0)
		customtkinter.CTkLabel(master=frame_map_ctrl, textvariable=self.canvas.variables['mousepos']['object'], text_font=("Terminal",9)).grid(row=5, column=1)

		customtkinter.CTkLabel(master=frame_map_ctrl,text="scanner coords:", text_font=("Terminal",9)).grid(row=6, column=0)
		customtkinter.CTkLabel(master=frame_map_ctrl, textvariable=self.variables['tippos']['object'], text_font=("Terminal",9)).grid(row=6, column=1)
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes

    
		return frame_map_ctrl




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

<<<<<<< Updated upstream
<<<<<<< Updated upstream
=======
=======
>>>>>>> Stashed changes


	def _canvas_compute_corners(self):

		self.canvas_size[0] = self.canvas.winfo_width()
		self.canvas_size[1] = self.canvas.winfo_height()

		# compute the canvas corner positions in physical space
		p = numpy.zeros(2)
		self.canvas_corners[0] = self.canvas_to_physical(p)
		p[0] = self.canvas_size[0]
		self.canvas_corners[1] = self.canvas_to_physical(p)
		p[1] = self.canvas_size[1]
		self.canvas_corners[2] = self.canvas_to_physical(p)
		p[0] = 0
		self.canvas_corners[3] = self.canvas_to_physical(p)
>>>>>>> Stashed changes


	def MoveTip(self, event):
		
		p = self.canvas.canvas_to_physical(numpy.asarray([event.x, event.y]))
		#print("canvas click at", event.x, event.y, "--",p)

		self.MoveTipFunction(p)
<<<<<<< Updated upstream
<<<<<<< Updated upstream

		# when the movement is done, show the position
		self._onTipPosChange(p)

=======
=======
>>>>>>> Stashed changes

		# when the movement is done, show the position
		self._onTipPosChange(p)

<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes

	def _onTipPosChange(self, newpos):

		p = newpos
		self.variables["tippos"]["value"][0] = p[0]
		self.variables["tippos"]["value"][1] = p[1]
		
		m0, u0, n0 = self.canvas.physical_to_approximate(p[0], 3)
		m1, u1, n1 = self.canvas.physical_to_approximate(p[1], 3)
		self.variables['tippos']['object'].set(
			"x:{:+.3f} {}, y:{:+.3f} {}".format(m0,u0, m1,u1)
		)
<<<<<<< Updated upstream
<<<<<<< Updated upstream
=======

		self.canvas.render()
>>>>>>> Stashed changes
=======

		self.canvas.render()
>>>>>>> Stashed changes

		self.canvas.render()


<<<<<<< Updated upstream
<<<<<<< Updated upstream
	'''
=======
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
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


<<<<<<< Updated upstream
<<<<<<< Updated upstream
=======
=======
>>>>>>> Stashed changes

>>>>>>> Stashed changes
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





<<<<<<< Updated upstream
<<<<<<< Updated upstream
=======
=======
>>>>>>> Stashed changes




<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======

>>>>>>> Stashed changes
class TabLithoPath(customtkinter.CTkFrame):

	def __init__(self, parent, controller):

		customtkinter.CTkFrame.__init__(self, parent)

		self.alanngui = controller
		self.frame_options_dict={} # when we load a GDS file, each shape will get its own frame that will
		# contain options to choose from on how to write. This dictionary will contain those frames
		self.gds = None

		self.grid_rowconfigure(0, weight=1)
		self.grid_columnconfigure(0, weight=0, minsize=240)
		self.grid_columnconfigure(1, weight=2, minsize=400)
<<<<<<< Updated upstream
<<<<<<< Updated upstream


		### panel variables
		self.variables = {
			'writefield': 	{'object': tk.StringVar(self), 'value':None},
			'writespeed': 	{'object': tk.StringVar(self), 'value':None},
			'idlespeed': 	{'object': tk.StringVar(self), 'value':None},
			'pitch': 		{'object': tk.StringVar(self), 'value':None},
			'exptype': 		{'object': tk.StringVar(self), 'value':None},
		}
		self.polygons = []
		self.lith_paths = []

		###########################
		# CANVAS FRAME
		###########################
		canvas = PhysicalCanvas(self, bg="white")
		canvas.grid(row=0, column=1,padx=8,pady=8, sticky="nsew")
		self.canvas = canvas

		canvas.callbacks['click'].append(self._canvas_onclick)

		#canvas.AddObject(CanvasLine("",numpy.asarray([[0,0],[5,5]]), fill="red"))
		#canvas.AddObject(CanvasPoint("",numpy.asarray([0,0]), pxsize=2, fill="blue"))

		
=======
=======
>>>>>>> Stashed changes


		###########################
		# frame with the right plot
		###########################
		#self.plotr = PlotFrame(self, parent, load=True)
		#self.plotr.grid(row=4,column=2, rowspan=2, sticky='nsew')
		##########################################
>>>>>>> Stashed changes

		#######################################
		# frame for path controls
		#######################################
		panel = customtkinter.CTkFrame(self)
		panel.grid(row=0,column=0, padx=8, pady=8, sticky='nsew')
		self.rast_prop = panel
		self.mainpanel = panel

		### title
		customtkinter.CTkLabel(panel, text="LithoPath Controls", text_font=('Roboto', 14)).grid(row=0, columnspan=3, pady=4, padx=10, sticky='n')

		### load file button
		self.gdsLoaded = False
		customtkinter.CTkButton(panel, text='Load file', command=self.openfile_onclick).grid(row=1,column=1)

<<<<<<< Updated upstream
<<<<<<< Updated upstream
		### control panel
		self.controlpanel = self._init_controlPanel(panel)
		self.controlpanel.grid(row=2, columnspan=3, padx=4,pady=4,sticky="new")
		
		
		### navigation panel
		self.navpanel = self._init_nav_panel(self.controlpanel)
		self.navpanel.grid(row=9, column=0, columnspan=2, padx=4, pady=4, sticky="new")
=======
=======
>>>>>>> Stashed changes

		
		### control panel
		self.controlpanel = self._init_controlPanel(panel)
		self.controlpanel.grid(row=2, columnspan=3, padx=4,pady=4,sticky="new")


		### canvas
		canvas = PhysicalCanvas(self, bg="white")
		canvas.grid(row=0, column=1,padx=8,pady=8, sticky="nsew")
		self.canvas = canvas

		#canvas.AddObject(CanvasLine("",numpy.asarray([[0,0],[5,5]]), fill="red"))
		#canvas.AddObject(CanvasPoint("",numpy.asarray([0,0]), pxsize=2, fill="blue"))


		# auto-resizing for frames within TabLithoPath (rast_prop and plotframe)
		#rows = [4,5]
		#columns = [2]
		#resizing(self, rows, columns)

<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes

	def _init_controlPanel(self, master):

		cp = customtkinter.CTkFrame(master)
		

		customtkinter.CTkLabel(cp, text="Raster settings").grid(row=0, column=0, columnspan=2, pady=4, sticky='n')


		customtkinter.CTkLabel(cp, text="Write Field Size [nm]: ").grid(row=1, column=0, pady=4, sticky='w')
<<<<<<< Updated upstream
<<<<<<< Updated upstream
		self.control_writefield = customtkinter.CTkEntry(cp, textvariable=self.variables['writefield']['object'])
		self.control_writefield.grid(row=1, column=1, padx=4, sticky='ew')
		
		customtkinter.CTkLabel(cp, text="Pitch [nm]: ").grid(row=2, column=0, pady=4, sticky='w')
		self.control_pitch = customtkinter.CTkEntry(cp, textvariable=self.variables['pitch']['object'])
		self.control_pitch.grid(row=2, column=1, padx=4, sticky='ew')

		customtkinter.CTkLabel(cp, text="Write Speed [nm/s]: ").grid(row=3, column=0, pady=4, sticky='w')		
		self.control_writespeed = customtkinter.CTkEntry(cp, textvariable=self.variables['writespeed']['object'])
		self.control_writespeed.grid(row=3, column=1, padx=4, sticky='ew')
		
		customtkinter.CTkLabel(cp, text="Idle Speed [nm/s]: ").grid(row=4, column=0, pady=4,sticky='w')
		self.control_idlespeed = customtkinter.CTkEntry(cp, textvariable=self.variables['idlespeed']['object'])
=======
=======
>>>>>>> Stashed changes
		self.tvar_writefield = tk.StringVar(cp)
		self.control_writefield = customtkinter.CTkEntry(cp, textvariable=self.tvar_writefield)
		self.control_writefield.grid(row=1, column=1, padx=4, sticky='ew')
		
		customtkinter.CTkLabel(cp, text="Pitch [nm]: ").grid(row=2, column=0, pady=4, sticky='w')
		self.tvar_pitch = tk.StringVar(cp)
		self.control_pitch = customtkinter.CTkEntry(cp, textvariable= self.tvar_pitch)
		self.control_pitch.grid(row=2, column=1, padx=4, sticky='ew')

		customtkinter.CTkLabel(cp, text="Write Speed [nm/s]: ").grid(row=3, column=0, pady=4, sticky='w')		
		self.tvar_writespeed = tk.StringVar(cp)
		self.control_writespeed = customtkinter.CTkEntry(cp, textvariable=self.tvar_writespeed)
		self.control_writespeed.grid(row=3, column=1, padx=4, sticky='ew')
		
		customtkinter.CTkLabel(cp, text="Idle Speed [nm/s]: ").grid(row=4, column=0, pady=4,sticky='w')
		self.tvar_idlespeed = tk.StringVar(cp)
		self.control_idlespeed = customtkinter.CTkEntry(cp, textvariable=self.tvar_idlespeed)
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
		self.control_idlespeed.grid(row=4, column=1, padx=4, sticky='ew')
		
		customtkinter.CTkCheckBox(cp, text="Invert image").grid(row=5, columnspan=2,pady=8, sticky='n')


		customtkinter.CTkLabel(cp, text="Export paths").grid(row=6, column=0, columnspan=2, pady=8, sticky='ew')

		# Entry fields and their labels
		customtkinter.CTkLabel(cp, text="Export as: ").grid(row=7, column=0, pady=4,sticky='w')

		options = ['Matrix Script','.txt file']
<<<<<<< Updated upstream
<<<<<<< Updated upstream
		self.control_exptype = ttk.OptionMenu(cp, self.variables['exptype']['object'], options[0], *options, command=self.exptype_onchange)
=======
		self.tvar_exptype = tk.StringVar(cp)
		self.control_exptype = ttk.OptionMenu(cp, self.tvar_exptype, options[0], *options, command=self.exptype_onchange)
>>>>>>> Stashed changes
=======
		self.tvar_exptype = tk.StringVar(cp)
		self.control_exptype = ttk.OptionMenu(cp, self.tvar_exptype, options[0], *options, command=self.exptype_onchange)
>>>>>>> Stashed changes
		self.control_exptype.grid(row=7, column=1, padx=4, sticky='e')

		customtkinter.CTkButton(cp, text='export', command=self.export_onclick).grid(row=8, columnspan=2, pady=4, sticky="n")

		return cp



<<<<<<< Updated upstream
<<<<<<< Updated upstream
	def _canvas_onclick(self, event):

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

				print("selected polygon",shapeID)
				selected = shapeID

				# mark as selected
				poly.options['fill'] = 'red'
				poly.options['width'] = 3
				self.frame_options_dict[shapeID].grid(row=10, columnspan=2, sticky='n')


				
			else:

				# deselect the polygon
				poly.options['fill'] = 'blue'
				poly.options['width'] = 1
				self.frame_options_dict[shapeID].grid_forget()

		self.canvas.render()
		


	def exptype_onchange(self, variable):

=======

	def exptype_onchange(self, variable):

>>>>>>> Stashed changes
=======

	def exptype_onchange(self, variable):

>>>>>>> Stashed changes
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


	def export_onclick(self):

		# takes in the shapes' coords and returns the vector coordinates for the scan. Should also replot with these vector coordinates
		# clear plot
		
		# define variables
		#write_field = int(self.tvar_write_field.get())
		pitch = int(self.variables['pitch']['object'].get())
		#write_speed = int(self.write_speed.get())
		#idle_speed = int(self.idle_speed.get())
<<<<<<< Updated upstream
<<<<<<< Updated upstream
=======
		self.shapes = {} #dictionary to hold all the shapes as 'shape' classes (from the GDSConverter.py file)
>>>>>>> Stashed changes
=======
		self.shapes = {} #dictionary to hold all the shapes as 'shape' classes (from the GDSConverter.py file)
>>>>>>> Stashed changes
		# get vector scan coordinates for each shape
		for shape in self.gds.shapes:
			write_type = self.frame_options_dict[shape].var_scan.get()
			scan_type = self.frame_options_dict[shape].var_fill.get() 
<<<<<<< Updated upstream
<<<<<<< Updated upstream
			self.gds.shapes[shape].vector_scan(write_type, scan_type, pitch)
=======
=======
>>>>>>> Stashed changes
			self.shapes[shape] = GDSConverter.Shape(self.plotr.content.shapes[shape]['coordinates'])
			self.shapes[shape].vector_scan(write_type, scan_type, pitch)
>>>>>>> Stashed changes
		# plot the new cooords
		# don't clear the canvas, just draw the raster path in a different colour so both
		# can be seen at the same time
		
		# remove any lithography paths currently on canvas
		for path in self.lith_paths: 
			self.canvas.RemoveObject(path.name)

		# draw the new lithography paths
		for shape in self.gds.shapes:
			lith_path = CanvasLine('lithpath'+str(shape), self.gds.shapes[shape].rasterPath, fill="red")
			self.canvas.AddObject(lith_path)
			self.lith_paths.append(lith_path)
		

	def openfile_onclick(self):


<<<<<<< Updated upstream
		file = filedialog.askopenfile(mode='r')
		if file:
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
			self.make_shape_frame(shapeID)


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
=======

	def openfile_onclick(self):


		file = filedialog.askopenfile(mode='r')
		if file:
			self.gds = GDSConverter.GDS(file)
			file.close()


<<<<<<< Updated upstream
=======

	def openfile_onclick(self):


		file = filedialog.askopenfile(mode='r')
		if file:
			self.gds = GDSConverter.GDS(file)
			file.close()


>>>>>>> Stashed changes
		# hopefully the file was opened and parsed correctly!


		'''
		# allows for file loading using file explorer window
		# child is the frame the plot is in that contains the dictionary with the shapes
		file = filedialog.askopenfile(mode='r')
		if file:
			child.content = GDSConverter.GDS(file)
			file.close()

		subplot.clear()

		for i in child.content.shapes:
			x = child.content.shapes[i]['coordinates'][:,0]
			y = child.content.shapes[i]['coordinates'][:,1]	
			subplot.plot(x,y, label="Shape {}".format(i))
			subplot.legend()
			self.make_shape_frame(i)
			canvaz.draw()
		'''
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes

	def make_shape_frame(self, n):
		# when we load up a design, each shape gets a panel with options on how to draw it. This makes the panels
		self.frame_options_dict[n] = shape_frame(self.controlpanel, n)
	
	def _init_nav_panel(self, mainframe):

		frame_map_ctrl = customtkinter.CTkFrame(master=mainframe, corner_radius=4)
		frame_map_ctrl.grid_columnconfigure(0, weight=0)
		frame_map_ctrl.grid_columnconfigure(1, weight=0, minsize=200)
		frame_map_ctrl.grid_columnconfigure(1, weight=1, minsize=20)

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


		
		return frame_map_ctrl



class shape_frame(customtkinter.CTkFrame):
	#######################
	# Frame which appears to let you select the different write options for the shapes
	#######################
	def __init__(self, parent, n):
		customtkinter.CTkFrame.__init__(self, parent)
		label = customtkinter.CTkLabel(self, text="Shape "+str(n), text_font=('Helvetica', 10)).grid(row=0, column=1, pady=5, padx=10, sticky='n')

		self.var_scan = tk.StringVar(self)
		options_scan = ['X-serpentine', 'Y-serpentine', 'Spiral', 'more tbc']
		WriteType = ttk.OptionMenu(self, self.var_scan, options_scan[0], *options_scan ).grid(row=1, column=2, pady=5,padx=10, sticky='e')
		WriteType_label = customtkinter.CTkLabel(self, text="Write type: ", text_font=('Helvetica', 10)).grid(row=1, column=1, pady=5, padx=10,sticky='w')

		self.var_fill = tk.StringVar(self)
		options_fill = ['Only fill', 'Fill and outline']
		ScanType = ttk.OptionMenu(self, self.var_fill, options_fill[0], *options_fill  ).grid(row=1, column=4, pady=5,padx=10, sticky='e')
		ScanType_label = customtkinter.CTkLabel(self, text="Scan type: ", text_font=('Helvetica', 10)).grid(row=1, column=3, pady=5, padx=10, sticky='w')







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
