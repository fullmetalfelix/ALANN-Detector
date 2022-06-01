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
import gds_conv # some custom classes/functions for importing and converting files (gds specifically atm) to vector coordinates for the tip



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



class ALANNGUI(customtkinter.CTk):
	# this is the main window which contains the different tabs and controls which is being seen


	def __init__(self, *args, **kwargs):
		customtkinter.CTk.__init__(self, *args, **kwargs)
		self.geometry("1200x800")
		container = customtkinter.CTkFrame(self)
		container.pack(side="top", fill="both", expand=True)
		self.title('ALANN') 
		resizing(container, [0], [2])

		# vertical side menu that allows navigation to different tabs
		menu = customtkinter.CTkFrame(container)
		self.menu_width=90
		HomeButton = customtkinter.CTkButton(menu, text="Home",
						command=lambda: self.show_frame(Home), width=self.menu_width)
		HomeButton.grid(row=1,column=1, sticky='n')
		RastPathButton = customtkinter.CTkButton(menu, text="Raster Path",
						command = lambda: self.show_frame(RastPath), width=self.menu_width)
		RastPathButton.grid(row=3,column=1, sticky='n')
		menu.grid(row=0,column=0,sticky='nsew')


		# dictionary containing all the tabs
		self.frames = {} 

		for F in (Home, RastPath):

			frame = F(container, self)

			self.frames[F] = frame

			frame.grid(row=0, column=2, sticky="nsew")
		
		self.show_frame(Home)

	def show_frame(self, cont): 
	# brings forward the frame of the tab you want to see
		frame = self.frames[cont]
		frame.tkraise()



class Home(customtkinter.CTkFrame):
	
	# max image size is 12 um
	# more values in between are needed
	# 1200nm 4.5um ...


	def __init__(self, parent, controller):	
		#ttk.Frame.__init__(self, parent)
		customtkinter.CTkFrame.__init__(self, parent)
		#self. = customtkinter.CTk()  # create CTk window like you do with the Tk window
		self._scans = []
	
		self.grid_rowconfigure(0, weight=1)
		self.grid_columnconfigure(0, weight=0, minsize=200)
		self.grid_columnconfigure(1, weight=2, minsize=400)

		# this should be the main control panel
		frame_ctrl = customtkinter.CTkFrame(master=self, width=250, height=240, corner_radius=4, name="controls")
		frame_ctrl.grid(row=0, column=0, padx=8, pady=8, sticky="nsew")
		self.frame_ctrl = frame_ctrl


	# this should be the main control panel
		frame_ctrl = customtkinter.CTkFrame(master=self, width=250, height=240, corner_radius=4, name="controls")
		frame_ctrl.grid(row=0, column=0, padx=8, pady=8, sticky="nsew")
		self.frame_ctrl = frame_ctrl


		customtkinter.CTkLabel(master=frame_ctrl,text="Controls").grid(row=0, column=0)


		customtkinter.CTkLabel(master=frame_ctrl,text="Imaging Parameters").grid(row=1, column=0)

		### image px size panel
		frm_pxsize = customtkinter.CTkFrame(master=frame_ctrl, name="pxsize")
		frm_pxsize.grid(row=2, column=0, padx=0, pady=2)

		customtkinter.CTkLabel(master=frm_pxsize,text="pixels:").grid(row=0, column=0)
		sld_px = customtkinter.CTkSlider(master=frm_pxsize, from_=6, to=11, number_of_steps=5,
			command=self.pxsize_change)
		sld_px.grid(row=0, column=1)
		tvar_pxsize = tk.StringVar(value="...")
		self.tvar_pxsize = tvar_pxsize
		lbl_pxsize = customtkinter.CTkLabel(master=frm_pxsize,textvariable=tvar_pxsize).grid(row=0, column=2)
		sld_px.set(8)
		self.sld_px = sld_px
		#########################

		### image physical size panel
		frm_size = customtkinter.CTkFrame(master=frame_ctrl, name="phsize")
		frm_size.grid(row=3, column=0, padx=0, pady=2)

		customtkinter.CTkLabel(master=frm_size,text="size:").grid(row=0, column=0)

		# from 100nm to 12um
		vals = [e for e in PhysicalSizes]
		sld_ph = customtkinter.CTkSlider(master=frm_size, from_=0, to=len(vals)-1, number_of_steps=len(vals)-1, 
			command=self.phsize_change)
		sld_ph.grid(row=0, column=1)
		tvar_phsize = tk.StringVar(value="...")
		self.tvar_phsize = tvar_phsize

		customtkinter.CTkLabel(master=frm_size,textvariable=tvar_phsize).grid(row=0, column=2)
		sld_ph.set(8)
		self.sld_ph = sld_ph
		#########################

		bt_scan = customtkinter.CTkButton(master=frame_ctrl, text="SCAN", command=self.scan_click)
		bt_scan.grid(row=4, column=0,pady=4)


		# and this is the map panel
		frame_map = customtkinter.CTkFrame(master=self, width=250, height=240, corner_radius=4)
		frame_map.grid(row=0, column=1, padx=8, pady=8, sticky="nsew")

		frame_map.grid_columnconfigure(0, weight=2)
		#frame_map.grid_columnconfigure(1, weight=0, minsize=20)
		frame_map.grid_rowconfigure(0, weight=2)
		#frame_map.grid_rowconfigure(1, weight=0, minsize=20)

		# canvas
		canvas = tk.Canvas(frame_map)
		canvas.grid(row=0, column=0,padx=4,pady=4, sticky="nsew")
		self.canvas = canvas

		### canvas navigation
		frame_map_ctrl = self._init_nav_panel(frame_ctrl)
		frame_map_ctrl.grid(row=5, column=0)
		
		# center of the canvas in physical space
		self.canvas_0 = numpy.asarray([0,0], dtype=numpy.float64)

		# canvas resolution in px/nm
		self.canvas_res = 100

		# canvas widget size in pixels
		self.canvas_size = numpy.zeros(2, dtype=numpy.int32)

		# canvas corner positions in physical space - order is ABCD clockwise A = top-left = canvas 0,0
		self.canvas_corners = numpy.zeros((4,2), dtype=numpy.float64)


		self._axisflipper = numpy.asarray([1,-1], dtype=numpy.float64)


		canvas.bind("<Configure>", self.resize)
		canvas.bind("<Button-1>", self.canvas_onclick)
		canvas.bind_all("<w>", self.canvas_onKeyPress)
		canvas.bind_all("<a>", self.canvas_onKeyPress)
		canvas.bind_all("<s>", self.canvas_onKeyPress)
		canvas.bind_all("<d>", self.canvas_onKeyPress)
		canvas.bind_all("<q>", self.canvas_onKeyPress)
		canvas.bind_all("<e>", self.canvas_onKeyPress)

	


		
		



	def _init_nav_panel(self, mainframe):

		frame_map_ctrl = customtkinter.CTkFrame(master=mainframe, corner_radius=4)
		
		customtkinter.CTkLabel(master=frame_map_ctrl,text="Navigation").grid(row=0, column=1)

		customtkinter.CTkButton(master=frame_map_ctrl, text="↑", command=self.bt_nav_up).grid(row=1, column=1,padx=4,pady=4)
		customtkinter.CTkButton(master=frame_map_ctrl, text="←", command=self.bt_nav_left).grid(row=2, column=0,padx=4,pady=4)
		customtkinter.CTkButton(master=frame_map_ctrl, text="→", command=self.bt_nav_right).grid(row=2, column=2,padx=4,pady=4)
		customtkinter.CTkButton(master=frame_map_ctrl, text="↓", command=self.bt_nav_down).grid(row=3, column=1,padx=4,pady=4)

		frame_map_zoom = customtkinter.CTkFrame(master=frame_map_ctrl, corner_radius=4)
		frame_map_zoom.grid(row=2, column=1)
		frame_map_zoom.grid_columnconfigure(0, weight=0, minsize=20)
		frame_map_zoom.grid_columnconfigure(1, weight=0, minsize=20)

		customtkinter.CTkButton(master=frame_map_zoom, text="+", width=40, command=self.bt_nav_zoomIN).grid(row=0,column=0, padx=4)
		customtkinter.CTkButton(master=frame_map_zoom, text="-", width=40, command=self.bt_nav_zoomOUT).grid(row=0,column=1, padx=4)
		

		customtkinter.CTkLabel(master=frame_map_ctrl,text="resolution:").grid(row=4, column=0)
		self.tvar_canvas_res = tk.StringVar(value="...")
		customtkinter.CTkLabel(master=frame_map_ctrl,
			textvariable=self.tvar_canvas_res).grid(row=4, column=1)

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
			u = "um"
		self.tvar_phsize.set("{} {}".format(s,u))

	def scan_click(self):

		print("scanning...")


		# calls Scanner.ScanImage(pixels, size, angle)
		vals = [e.value for e in PhysicalSizes]
		size = vals[int(self.sld_ph.get())]
		npx = int(numpy.power(2, self.sld_px.get()))
		
		scan = self.ScanFunction(npx, size, 0)
		self._scans.append(scan)
		print("scan completed")

		#plt.matshow(scan.data)
		#plt.show()


		self.canvas_redraw()


	def bt_nav_up(self): self.canvas_move([0,-1])
	def bt_nav_down(self): self.canvas_move([0,1])
	def bt_nav_left(self): self.canvas_move([-1,0])
	def bt_nav_right(self): self.canvas_move([1,0])
	def canvas_move(self, direction):

		step = 0.1 * self.canvas_size / self.canvas_res
		self.canvas_0 += numpy.asarray(direction) * self._axisflipper * step
		self.canvas_redraw()
	def bt_nav_zoomIN(self):

		self.canvas_res *= 1.5
		self.canvas_redraw()
	def bt_nav_zoomOUT(self):

		self.canvas_res /= 1.5
		self.canvas_redraw()


	def canvas_onKeyPress(self, event):
		#print("pressed", event.char)
		if event.char == "w": 	self.bt_nav_up()
		elif event.char == "s":	self.bt_nav_down()
		elif event.char == "a":	self.bt_nav_left()
		elif event.char == "d":	self.bt_nav_right()
		elif event.char == "q":	self.bt_nav_zoomIN()
		elif event.char == "e":	self.bt_nav_zoomOUT()


	def resize(self, event):

		self.canvas_redraw()
		
	
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


	def canvas_onclick(self,event):
		
		p = self.canvas_to_physical(numpy.asarray([event.x, event.y]))

		print("canvas click at", event.x, event.y, "--",p)

		self.MoveTipFunction(p)
		self.canvas_redraw()


	## converts coordinates from physical space into canvas pixel space
	def physical_to_canvas(self, point):

		v = self.canvas_size * 0.5
		v += self._axisflipper * (point - self.canvas_0) * self.canvas_res
		return v

	def canvas_to_physical(self, pxpoint):

		v = self._axisflipper * pxpoint
		v-= self._axisflipper * self.canvas_size*0.5
		v/= self.canvas_res
		v+= self.canvas_0

		return v

	def canvas_redraw(self):

		self.canvas.delete("all")

		self._canvas_compute_corners()

		self._pixelBuffer = numpy.zeros(self.canvas_size, dtype=numpy.float64)
		self._pixelStats = numpy.zeros(self.canvas_size, dtype=numpy.int32)

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

		'''
		self._pixelStats[self._pixelStats==0] = 1
		self._pixelBuffer /= self._pixelStats
		self._pixelBuffer = self._pixelBuffer.astype(numpy.uint8)
		pic = Image.fromarray(self._pixelBuffer)
		if self._pixelBuffer.shape[0] > 100:
			pic.save("canvas.png", format="PNG")
		img =  ImageTk.PhotoImage(image=pic)
		self.canvas.create_image(0,0, anchor="nw", image=img)
		'''



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

		# if both corners of an edge are on the same side of the canvas, the image is out
		spm_x0 = spm.x_offset
		spm_xm = spm.x_offset + spm.width

		frame_x0 = numpy.max([x0,spm_x0])
		frame_xm = numpy.min([xm,spm_xm])

		#print("frame x",frame_x0,frame_xm,x0,xm)

		if frame_xm < x0 or frame_x0 > xm:
			#print("spm is out of canvas (x)")
			return None

		spm_y0 = spm.y_offset
		spm_ym = spm.y_offset + spm.height

		frame_y0 = numpy.max([y0,spm_y0])
		frame_ym = numpy.min([ym,spm_ym])

		if frame_ym < y0 or frame_y0 > ym:
			#print("spm is out of canvas (y)")
			return None

		# code here => there is some overlap between spm and canvas
		#print("frame boundaries on spm (x):",[spm_x0,frame_x0],[spm_xm,frame_xm])
		#print("frame boundaries on spm (y):",[spm_y0,frame_y0],[spm_ym,frame_ym])

		# crop the original spm data 
		data = spm.data - self._imgmin # also applies the shift
		data /= self._imgmax*0.5
		data *= 255


		# where is frame_x0 in spm pixel coordinates?
		frame_px_x0 = int(numpy.floor((frame_x0 - spm_x0) / spm.pixelSize[0]))
		frame_px_xm = int((frame_xm-spm_x0) / spm.pixelSize[0])
		if frame_px_xm == 0: frame_px_xm = 1

		#print("frame pixel coords (x)",frame_px_x0,frame_px_xm)
		data = data[:,frame_px_x0:frame_px_xm+1]

		frame_px_y0 = int(numpy.floor((frame_y0 - spm_y0) / spm.pixelSize[1]))
		frame_px_ym = int((frame_ym-spm_y0) / spm.pixelSize[1])
		if frame_px_ym == 0: frame_px_ym = 1

		#print("frame pixel coords (y)",frame_px_y0,frame_px_ym)
		data = data[frame_px_y0:frame_px_ym+1]

		crop_px = numpy.asarray([data.shape[1],data.shape[0]])
		crop_nm = crop_px * spm.pixelSize
		#print("cropped size {}px - {}nm".format(crop_px, crop_nm))

		#print("data stats",numpy.mean(spm.data),numpy.min(spm.data),numpy.max(spm.data))

		'''
		#imgmin = numpy.min(spm.data)
		data = spm.data - self._imgmin
		data /= self._imgmax*0.5
		data *= 255
		'''
		# final conversion to bytes and flip vertically
		data = data.astype(numpy.uint8)
		data = numpy.flip(data, axis=0)


		# we have to make the spm pixels the same size as the canvas pixels
		# canvas pixel size is 1 / self.canvas_res
		# spm pixel size is spm.pixelSize (x,y components)

		trgPXsize = numpy.asarray([1,1]) / self.canvas_res
		curPXsize = spm.pixelSize
		scaling = curPXsize / trgPXsize
		newsize = numpy.round(numpy.asarray([data.shape[1],data.shape[0]]) * scaling)
		newsize = newsize.astype(numpy.uint32)
		method = Image.Resampling.BICUBIC
		if scaling[0] < 1 and scaling[1] < 1:
			method = Image.Resampling.LANCZOS
		#print(trgPXsize,curPXsize,scaling,"--",data.shape, newsize)

		
		pic = Image.fromarray(data)
		pic = pic.resize(newsize, resample=method)
		tkpic = ImageTk.PhotoImage(image=pic)
		#pic.save("canvas.png", format="PNG")
		self._crops.append(tkpic)



		p = numpy.asarray([frame_x0, frame_y0], dtype=numpy.float64)
		c = self.physical_to_canvas(p)
		#print("canvas placement:",p,c)
		self.canvas.create_image(c[0],c[1], image=tkpic, anchor="sw")
		#print("the spm is now {}w x {}h [nm]".format(pic.size[0]/self.canvas_res, pic.size[1]/self.canvas_res))

		return


	def canvas_redraw_scalebar(self):

		cw = self.canvas.winfo_width()
		ch = self.canvas.winfo_height()

		# draw the scale bar
		barsize_px = 0.1 * cw # bar size in pixels - how many nm is that?
		barsize_nm = barsize_px / self.canvas_res # size in nm -> round it
		barsize_nm = numpy.round(barsize_nm) # then get the fixed pixel count
		barsize_px = numpy.round(barsize_nm * self.canvas_res)
		bartxt = "{} nm".format(barsize_nm)
		if barsize_nm > 1000: bartxt = "{} um".format(barsize_nm/1000)
		self.canvas.create_rectangle(cw-20-barsize_px, ch-20-10, cw-20, ch-20, fill="black",outline="white", width=2)
		self.canvas.create_rectangle(cw-20-2*barsize_px, ch-20-10+2, cw-20-barsize_px, ch-20-2, fill="white",outline="black", width=2)
		self.canvas.create_text(cw-20-barsize_px/2, ch-10, justify=tk.CENTER, text=bartxt)


	def canvas_redraw_tip(self):

		tip = self.GetTipFunction()
		ctip = self.physical_to_canvas(tip)

		
		self.canvas.create_line(ctip[0], ctip[1]-8, ctip[0], ctip[1]-2, fill="red")
		self.canvas.create_line(ctip[0], ctip[1]+8, ctip[0], ctip[1]+2, fill="red")

		self.canvas.create_line(ctip[0]-8, ctip[1], ctip[0]-2, ctip[1], fill="red")
		self.canvas.create_line(ctip[0]+8, ctip[1], ctip[0]+2, ctip[1], fill="red")

class RastPath(customtkinter.CTkFrame):

	def __init__(self, parent, controller):
		customtkinter.CTkFrame.__init__(self,parent)

		self.frame_options_dict={} # when we load a GDS file, each shape will get its own frame that will
		# contain options to choose from on how to write. This dictionary will contain those frames

		# title
		title = customtkinter.CTkLabel(self, text="Raster Path Determination", text_font = ("Helvetica",33) )
		title.grid(row=0,column=1, columnspan=3)
		description = customtkinter.CTkLabel(self, text = "Load a file (.txt, .mat, .bmap,...)", text_font = ("Helvetica",15))
		description.grid(row=1, column=1, columnspan=3)

		############################
		# frame with the right plot
		###########################
		self.plotr = PlotFrame(self, parent, load=True)
		self.plotr.grid(row=4,column=2, rowspan=2, sticky='nsew')
		##########################################

		#######################################
		# frame for Raster Properties
		#######################################
		self.rast_prop = customtkinter.CTkFrame(self)
		self.rast_prop.grid(row=4,column=1, sticky='nsew')
		layer_label = customtkinter.CTkLabel(self.rast_prop, text="Raster Properties", text_font=('Helvetica', 15)).grid(row=0, columnspan=2, pady=5, padx=10, sticky='ew')

		# Entry fields and their labels
		options = ['Matrix Script','.txt file']
		self.var_type = tk.StringVar(self.rast_prop)
		ExportAsType = ttk.OptionMenu(self.rast_prop, self.var_type, options[0], *options, command = self.change_rast_prop ).grid(row=1, column=1, pady=5, padx=10, sticky='ew')
		ExportAsType_label = customtkinter.CTkLabel(self.rast_prop, text="Export as: ", text_font=('Helvetica', 10)).grid(row=1, column=0, pady=5, padx=10,sticky='ew')

		self.write_field = tk.StringVar(self.rast_prop)
		self.writeFieldSize = customtkinter.CTkEntry(self.rast_prop, textvariable=self.write_field)
		self.writeFieldSize.grid(row=5, column=1,pady=5,  padx=10,sticky='ew')
		writeFieldSize_label = customtkinter.CTkLabel(self.rast_prop, text="Write Field Size [nm]: ", text_font=('Helvetica', 10)).grid(row=5, column=0, pady=5, padx=10,sticky='ew')

		self.pitch = tk.StringVar(self.rast_prop)
		Pitch = customtkinter.CTkEntry(self.rast_prop, textvariable= self.pitch).grid(row=2, column=1,pady=5, padx=10, sticky='ew')
		Pitch_label = customtkinter.CTkLabel(self.rast_prop, text="Pitch [nm]: ", text_font=('Helvetica', 10)).grid(row=2, column=0, pady=5, padx=10,sticky='ew')

		self.write_speed = tk.StringVar(self.rast_prop)
		self.WriteSpeed = customtkinter.CTkEntry(self.rast_prop, textvariable=self.write_speed)
		self.WriteSpeed.grid(row=6, column=1, pady=5, padx=10, sticky='ew')
		WriteSpeed_label = customtkinter.CTkLabel(self.rast_prop, text="Write Speed [nm/s]: ", text_font=('Helvetica', 10)).grid(row=6, column=0, pady=5, padx=10, sticky='ew')

		self.idle_speed = tk.StringVar(self.rast_prop)
		self.IdleSpeed = customtkinter.CTkEntry(self.rast_prop, textvariable=self.idle_speed)
		self.IdleSpeed.grid(row=7, column=1,pady=5, padx=10, sticky='ew')
		IdleSpeed_label = customtkinter.CTkLabel(self.rast_prop, text="Idle Speed [nm/s]: ", text_font=('Helvetica', 10)).grid(row=7, column=0, pady=5, padx=10,sticky='ew')

		InvertImg = customtkinter.CTkCheckBox(self.rast_prop,text="Invert Image?").grid(row=8, column=0, columnspan=2,pady=5,padx=10, sticky='n')

		ConvRastPathButton = customtkinter.CTkButton(self.rast_prop, text='Convert and Export Raster Paths', command = lambda: self.convert() )
		ConvRastPathButton.grid(row=10, columnspan=2, pady=5 , padx=10)

		#######################################

		# auto-resizing for frames within RastPath (rast_prop and plotframe)
		rows = [4,5]
		columns = [2]
		resizing(self, rows, columns)

	def change_rast_prop(self,variable):
		variable = self.var_type.get()
		if variable=='.txt file':
			self.writeFieldSize.config(state=tk.DISABLED)
			self.WriteSpeed.config(state=tk.DISABLED)
			self.IdleSpeed.config(state=tk.DISABLED)
		if variable=='Matrix Script':
			self.writeFieldSize.config(state=tk.NORMAL)
			self.WriteSpeed.config(state=tk.NORMAL)
			self.IdleSpeed.config(state=tk.NORMAL)

	def convert(self):
	# takes in the shapes' coords and returns the vector coordinates for the scan. Should also replot with these vector coordinates
		# clear plot
		self.plotr.subplot.clear()
		# define variables
		#write_field = int(self.write_field.get())
		pitch = int(self.pitch.get())
		#write_speed = int(self.write_speed.get())
		#idle_speed = int(self.idle_speed.get())
		self.shapes = {} #dictionary to hold all the shapes as 'shape' classes (from the gds_conv.py file)
		# get vector scan coordinates for each shape
		for shape in self.plotr.content.shapes:
			write_type = self.frame_options_dict[shape].var_scan.get()
			scan_type = self.frame_options_dict[shape].var_fill.get() 
			self.shapes[shape] = gds_conv.shape(self.plotr.content.shapes[shape]['coordinates'])
			self.shapes[shape].vector_scan(write_type, scan_type, pitch)
		# plot the new cooords
		self.plotr.update_plot(self.shapes)
		self.plotr.canvaz.draw()

	def open_file(self, child, subplot, canvaz):
		# allows for file loading using file explorer window
		# child is the frame the plot is in that contains the dictionary with the shapes
		file = filedialog.askopenfile(mode='r')
		if file:
			child.content = gds_conv.GDS_file(file)
			file.close()
		subplot.clear()
		for i in child.content.shapes:
			x = child.content.shapes[i]['coordinates'][:,0]
			y = child.content.shapes[i]['coordinates'][:,1]	
			subplot.plot(x,y, label="Shape "+str(i))
			subplot.legend()
			self.make_shape_frame(i)
			canvaz.draw()

	def make_shape_frame(self, n):
		# when we load up a design, each shape gets a panel with options on how to draw it. This makes the panels
		self.frame_options_dict[n] = shape_frame(self, n)
		resizing(self, [5+n],[])
		self.plotr.grid(rowspan=n+2)

	def clear(self,subplot, canvas):
		# gets rid of everything in the plot and deletes the panels made by make_shape_frame
		subplot.clear()
		canvas.draw()
		for i in self.frame_options_dict:
			self.frame_options_dict[i].grid_forget()
			self.frame_options_dict[i].destroy()


class shape_frame(customtkinter.CTkFrame):
	#######################
	# Frame which appears to let you select the different write options for the shapes
	#######################
	def __init__(self, parent, n):
		customtkinter.CTkFrame.__init__(self, parent)
		self.grid(row=5+n,column=1, sticky='nesw')
		label = customtkinter.CTkLabel(self, text="Shape "+str(n), text_font=('Helvetica', 10)).grid(row=0, column=1, pady=5, padx=10, sticky='n')

		self.var_scan = tk.StringVar(self)
		options_scan = ['X-serpentine', 'Y-serpentine', 'Spiral', 'more tbc']
		WriteType = ttk.OptionMenu(self, self.var_scan, options_scan[0], *options_scan ).grid(row=1, column=2, pady=5,padx=10, sticky='e')
		WriteType_label = customtkinter.CTkLabel(self, text="Write type: ", text_font=('Helvetica', 10)).grid(row=1, column=1, pady=5, padx=10,sticky='w')

		self.var_fill = tk.StringVar(self)
		options_fill = ['Only fill', 'Fill and outline']
		ScanType = ttk.OptionMenu(self, self.var_fill, options_fill[0], *options_fill  ).grid(row=1, column=4, pady=5,padx=10, sticky='e')
		ScanType_label = customtkinter.CTkLabel(self, text="Scan type: ", text_font=('Helvetica', 10)).grid(row=1, column=3, pady=5, padx=10, sticky='w')



class PlotFrame(customtkinter.CTkFrame):
	###########################
	# frame with a matplotlib plot
	###########################
	def __init__(self, parent, controller, load=False):
		customtkinter.CTkFrame.__init__(self, parent, width=150, height=150) 
		#make our figure and add blank plot
		f = Figure(figsize=(2,3), dpi=100)
		self.subplot = f.add_subplot(111)
		self.content = None

		self.canvaz = FigureCanvasTkAgg(f, self) 
		self.canvaz.draw()
		self.canvaz.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10,pady=10)

		toolbar = NavigationToolbar2Tk(self.canvaz, self)
		toolbar.update()
		self.canvaz._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=20,pady=10)
		#button to clear the canvaz
		clear = customtkinter.CTkButton(self, text='Clear', command = lambda: parent.clear(self.subplot, self.canvaz) )
		clear.pack(side=tk.LEFT,padx=5,pady=5)

		if load:
			load = customtkinter.CTkButton(self, text='Load', command = lambda: parent.open_file(self, self.subplot, self.canvaz) ) 
			#makes a button that carries out the open_file function when clicked
			load.pack(side=tk.LEFT,padx=5,pady=5)
			# method that allows you to browse

	def update_plot(self, shapes):
		for shape in shapes:
			x = shapes[shape].coords[:,0]
			y = shapes[shape].coords[:,1]	
			self.subplot.plot(x,y,linewidth=0.5, label='Shape '+str(shape))
		self.subplot.legend()
		self.canvaz.draw()	





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
	gui.frames[Home].ScanFunction = scn.ScanImage
	gui.frames[Home].MoveTipFunction = scn.MoveTip
	gui.frames[Home].GetTipFunction = scn.GetTip


	# run the app
	gui.mainloop()
