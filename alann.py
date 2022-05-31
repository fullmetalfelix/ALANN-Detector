import tkinter
import customtkinter
import numpy
from enum import IntEnum
from scanner import Sample, SampleCheck, Scanner, SPM
import PIL
from PIL import Image, ImageTk
import matplotlib.pyplot as plt



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


class ALANNGUI(object):


	# max image size is 12 um
	# more values in between are needed
	# 1200nm 4.5um ...


	def __init__(self):

		self._scans = []

		root_tk = customtkinter.CTk()  # create CTk window like you do with the Tk window
		root_tk.geometry("1200x800")
		root_tk.title("ALANN")


		root_tk.grid_rowconfigure(0, weight=1)
		root_tk.grid_columnconfigure(0, weight=0, minsize=200)
		root_tk.grid_columnconfigure(1, weight=2, minsize=400)


		# this should be the main control panel
		frame_ctrl = customtkinter.CTkFrame(master=root_tk, width=250, height=240, corner_radius=4, name="controls")
		frame_ctrl.grid(row=0, column=0, padx=8, pady=8, sticky="nsew")
		self.frame_ctrl = frame_ctrl




		customtkinter.CTkLabel(master=frame_ctrl,text="Controls").grid(row=0, column=0)

		frm_scan = self._init_scan_panel(frame_ctrl)
		frm_scan.grid(row=1, column=0, pady=4)

		
		nrow = 1



		# and this is the map panel
		frame_map = customtkinter.CTkFrame(master=root_tk, width=250, height=240, corner_radius=4)
		frame_map.grid(row=0, column=1, padx=8, pady=8, sticky="nsew")

		frame_map.grid_columnconfigure(0, weight=2)
		#frame_map.grid_columnconfigure(1, weight=0, minsize=20)
		frame_map.grid_rowconfigure(0, weight=2)
		#frame_map.grid_rowconfigure(1, weight=0, minsize=20)

		# canvas
		canvas = tkinter.Canvas(frame_map)
		canvas.grid(row=0, column=0,padx=4,pady=4, sticky="nsew")
		self.canvas = canvas

		### canvas navigation
		frame_map_ctrl = self._init_nav_panel(frame_ctrl)
		frame_map_ctrl.grid(row=nrow+1, column=0)
		
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
		canvas.bind('<Motion>', self._canvas_onMouseMove)
		canvas.bind('<Leave>', self._canvas_onMouseOut)
		canvas.bind_all("<w>", self.canvas_onKeyPress)
		canvas.bind_all("<a>", self.canvas_onKeyPress)
		canvas.bind_all("<s>", self.canvas_onKeyPress)
		canvas.bind_all("<d>", self.canvas_onKeyPress)
		canvas.bind_all("<q>", self.canvas_onKeyPress)
		canvas.bind_all("<e>", self.canvas_onKeyPress)

		self.root_tk = root_tk


		
		

	def _init_scan_panel(self, mainframe):

		frm_scan = customtkinter.CTkFrame(master=mainframe)

		# title label
		customtkinter.CTkLabel(master=frm_scan,text="Imaging Parameters").grid(row=0, column=0)

		# slider panels
		frm_pxsize = customtkinter.CTkFrame(master=frm_scan) #, fg_color="#FF0000")
		frm_phsize = customtkinter.CTkFrame(master=frm_scan)
		frm_angle = customtkinter.CTkFrame(master=frm_scan, name="phangle")

		frm_pxsize.grid(row=1, column=0, padx=0, pady=2)
		frm_phsize.grid(row=2, column=0, padx=0, pady=2)
		frm_angle.grid(row=3, column=0, padx=0, pady=2)



		# image px size panel

		customtkinter.CTkLabel(master=frm_pxsize,text="pixels:").grid(row=0, column=0)

		sld_px = customtkinter.CTkSlider(master=frm_pxsize, from_=6, to=11, number_of_steps=5, command=self.pxsize_change)
		sld_px.grid(row=0, column=1)
		self.sld_px = sld_px

		tvar_pxsize = tkinter.StringVar(value="...")
		self.tvar_pxsize = tvar_pxsize
		lbl_pxsize = customtkinter.CTkLabel(master=frm_pxsize,textvariable=tvar_pxsize).grid(row=0, column=2)
		
		sld_px.set(8)
		


		# img physical size

		customtkinter.CTkLabel(master=frm_phsize,text="size:").grid(row=0, column=0)

		# from 100nm to 12um
		vals = [e for e in PhysicalSizes]
		sld_ph = customtkinter.CTkSlider(master=frm_phsize, from_=0, to=len(vals)-1, number_of_steps=len(vals)-1, command=self.phsize_change)
		sld_ph.grid(row=0, column=1)
		self.sld_ph = sld_ph


		tvar_phsize = tkinter.StringVar(value="...")
		self.tvar_phsize = tvar_phsize

		customtkinter.CTkLabel(master=frm_phsize,textvariable=tvar_phsize).grid(row=0, column=2)
		sld_ph.set(8)
		


		# img fast-scan angle

		customtkinter.CTkLabel(master=frm_angle,text="angle:").grid(row=0, column=0)
		
		sld_angle = customtkinter.CTkSlider(master=frm_angle, from_=-90, to=90, number_of_steps=180, command=self.phang_change)
		sld_angle.grid(row=0, column=1)
		self.sld_angle = sld_angle

		tvar_phang = tkinter.StringVar(value="...")
		self.tvar_phang = tvar_phang
		customtkinter.CTkLabel(master=frm_angle,textvariable=tvar_phang).grid(row=0, column=2)

		
		sld_angle.set(0)
		
		bt_scan = customtkinter.CTkButton(master=frm_scan, text="SCAN", command=self.scan_click)
		bt_scan.grid(row=4, column=0,pady=4)

		return frm_scan


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
		

		customtkinter.CTkLabel(master=frame_map_ctrl,text="resolution:", text_font=("Terminal",9)).grid(row=4, column=0)
		self.tvar_canvas_res = tkinter.StringVar(value="...")
		customtkinter.CTkLabel(master=frame_map_ctrl, textvariable=self.tvar_canvas_res, text_font=("Terminal",9)).grid(row=4, column=1)

		customtkinter.CTkLabel(master=frame_map_ctrl,text="mouse coords:", text_font=("Terminal",9)).grid(row=5, column=0)
		self.tvar_canvas_mouse = tkinter.StringVar(value="...")
		customtkinter.CTkLabel(master=frame_map_ctrl, textvariable=self.tvar_canvas_mouse, text_font=("Terminal",9)).grid(row=5, column=1)

		customtkinter.CTkLabel(master=frame_map_ctrl,text="scanner coords:", text_font=("Terminal",9)).grid(row=6, column=0)
		self.tvar_canvas_scanner = tkinter.StringVar(value="...")
		customtkinter.CTkLabel(master=frame_map_ctrl, textvariable=self.tvar_canvas_scanner, text_font=("Terminal",9)).grid(row=6, column=1)



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

	def phang_change(self,value):

		self.tvar_phang.set("{} deg".format(value))


	def scan_click(self):

		print("scanning...")


		# calls Scanner.ScanImage(pixels, size, angle)
		vals = [e.value for e in PhysicalSizes]
		size = vals[int(self.sld_ph.get())]
		npx = int(numpy.power(2, self.sld_px.get()))
		angle = self.sld_angle.get()

		scan = self.ScanFunction(npx, size, angle)
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


	def _canvas_onMouseMove(self, event):
		
		x, y = event.x, event.y
		c = numpy.asarray([x,y])
		p = self.canvas_to_physical(c)
		u = ["nm", "nm"]
		for i in range(2):
			if numpy.abs(p[i]) > 1000:
				p[i] /= 1000
				u[i] = "um"

		self.tvar_canvas_mouse.set("{:+.3f} {}, {:+.3f} {}".format(p[0],u[0],p[1],u[1]))


	def _canvas_onMouseOut(self, event):

		self.tvar_canvas_mouse.set("---")


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

		'''
		pic = Image.fromarray(data)
		pic = pic.resize(newsize, resample=method)
		pic.convert('RGBA')
		tkpic = ImageTk.PhotoImage(image=pic)
		pic.save("canvas.png", format="PNG")

		pic2 = pic.rotate(30, expand=True, center=(0,1))

		mask = numpy.zeros(data.shape,dtype=numpy.uint8)
		mask += 255
		mask = Image.fromarray(mask)
		mask = mask.rotate(30, expand=True)
		mask.save("canvas.mask.png", format="PNG")

		bgim = numpy.zeros((data.shape[0],data.shape[1],4),dtype=numpy.uint8)
		bgim[:,:,0] = bgim[:,:,1] = bgim[:,:,2] = 255
		bgim = Image.fromarray(bgim, mode="RGBA")
		bgim = bgim.rotate(30, expand=True)
		bgim.save("canvas.bgim.png", format="PNG")

		#rotm = Image.composite(pic2, fff, mask)
		#fff = Image.new('RGBA', pic2.size, (255,)*4)
		#pic2 = Image.composite(pic2, fff, pic2)
		pic2.save("canvas.rot.png", format="PNG")
		'''
		'''# original image
		img = Image.open('test.png')
		# converted to have an alpha layer
		im2 = img.convert('RGBA')
		# rotated image
		rot = im2.rotate(22.2, expand=1)
		# a white image same size as rotated image
		fff = Image.new('RGBA', rot.size, (255,)*4)
		# create a composite image using the alpha layer of rot as a mask
		out = Image.composite(rot, fff, rot)
		# save your work (converting back to mode='1' or whatever..)
		out.convert(img.mode).save('test2.bmp')
		'''


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
		self.canvas.create_text(cw-20-barsize_px/2, ch-10, justify=tkinter.CENTER, text=bartxt)


	# makes the crosshair at the scanner position
	def canvas_redraw_tip(self):

		tip = self.GetTipFunction()
		ctip = self.physical_to_canvas(tip)
		u = ["nm","nm"]
		for i in range(2):
			if numpy.abs(tip[i]) > 1000:
				tip[i] /= 1000
				u[i] = "um"

		self.tvar_canvas_scanner.set("x:{:+.3f} {}, y:{:+.3f} {}".format(tip[0],u[0],tip[1],u[1]))
		
		self.canvas.create_line(ctip[0], ctip[1]-8, ctip[0], ctip[1]-2, fill="red")
		self.canvas.create_line(ctip[0], ctip[1]+8, ctip[0], ctip[1]+2, fill="red")

		self.canvas.create_line(ctip[0]-8, ctip[1], ctip[0]-2, ctip[1], fill="red")
		self.canvas.create_line(ctip[0]+8, ctip[1], ctip[0]+2, ctip[1], fill="red")



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
	gui.ScanFunction = scn.ScanImage
	gui.MoveTipFunction = scn.MoveTip
	gui.GetTipFunction = scn.GetTip


	# run the app
	gui.root_tk.mainloop()
