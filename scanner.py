### this is a fake scanner that generates a sample topography using perlin noise

import numpy
import pickle
import matplotlib.pyplot as plt


class Sample(object):



	def __init__(self, size, scale, persist, luna, noct):

		# sample size in nm
		self.size = size
		self.scale = scale

		# number of octaves
		self.noct = noct
		self.luna = luna
		self.persist = persist

		self.octaves = []
		self.olen = numpy.zeros(self.noct, dtype=numpy.int64)
		n = 2
		fn = 2
		for o in range(self.noct):

			octave = scale*numpy.random.rand(n+1,n+1)
			octave[-1] = octave[0]
			self.octaves.append(octave)
			self.olen[o] = n

			fn *= luna
			n = int(fn)

		# physical step size is 0.12 nm


	def Save(self, filename):

		d = {
			'size': self.size,
			'scale': self.scale,
			'persist': self.persist,
			'luna': self.luna,
			'octaves': self.octaves,
		}

		with open(filename, 'wb') as handle:
			pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

	def Load(filename):

		d = {}
		with open(filename, 'rb') as handle:
			d = pickle.load(handle)

		noct = len(d['octaves'])
		sample = Sample(d['size'], d['scale'], d['persist'], d['luna'], noct)
		sample.octaves = d['octaves']

		return sample


	def Topography(self,x,y):

		# x,y are in nm
		# output z is also in nm

		# fractional coordinates
		xr = numpy.asarray([x,y])
		xr = xr - numpy.floor(xr/self.size)*self.size

		#xt = x / self.size; xt = xt - numpy.floor(xt)
		#yt = y / self.size; yt = yt - numpy.floor(yt)
		#tpos = numpy.asarray([xt,yt])
		tpos = xr
		p = 1.0
		z = 0


		for o in range(self.noct):
			
			oc = self.octaves[o]
			m = self.olen[o]
			t = xr*m / self.size
			t = t-numpy.floor(t/m)*m
			xi = xr*m/self.size
			xi = numpy.floor(xi).astype(numpy.int32)
			yi = xi+1
			yi = yi-numpy.floor(yi/m).astype(numpy.int32)*m
			t = t-xi
			xi+=1;yi+=1


			s1 = oc[xi[0],xi[1]]*(1-t[0]) + oc[yi[0],xi[1]]*t[0]
			s2 = oc[xi[0],yi[1]]*(1-t[0]) + oc[yi[0],yi[1]]*t[0]
			s = s1*(1-t[1]) + s2*t[1]
			
			z += p*s;
			p *= self.persist
			
		
		z = numpy.floor(z / 0.136) * 0.136
		return z


class SampleCheck(Sample):

	def __init__(self, boxsize):

		self.boxsize = boxsize


	def Topography(self,x,y):

		# x,y are in nm
		# output z is also in nm

		z = numpy.sin(2*numpy.pi*x/self.boxsize) * numpy.sin(2*numpy.pi*y/self.boxsize)

		return z



## Base class for SPM images
class SPM(object):

	
	## Create an SPM image from pixel matrix and other parameters
	# \param data: pixel matrix as numpy array
	# \param width: image width in meters
	# \param height: image height in meters
	def __init__(self, data, width, height, y_offset, x_offset, angle, channel_name_and_unit=['','']):

		self.data = data
		self.width = width
		self.height = height
		self.y_offset = y_offset
		self.x_offset = x_offset
		self.angle = angle
		self.channel_name_and_unit = channel_name_and_unit

		self.processing = {}

		
		# size of one pixel (along x,y) in physical space
		self.pixelSize = numpy.asarray([height,width]) / self.data.shape
		self.pixelSize = numpy.flip(self.pixelSize)

		# if the image has an angle, compute the corrected frame coordinates

		# corrected phsical coords of the corners
		corners = numpy.zeros((4,2),dtype=numpy.float64)

		theta = self.angle * numpy.pi / 180
		rmat = numpy.asarray([[numpy.cos(theta),-numpy.sin(theta)],[numpy.sin(theta),numpy.cos(theta)]])
		cmat = [[self.width,self.width,0],[0,self.height,self.height]]
		cmat = numpy.asarray(cmat)
		cmat = numpy.matmul(rmat, cmat) # cmat has the corner coords on the columns
		cmat = numpy.transpose(cmat) # and now they are on the rows

		corners[1:] = cmat
		corners[:,0] += self.x_offset
		corners[:,1] += self.y_offset

		# corner positions (physical space)
		self.image_corners = corners

		# now the image frame with the axes alined to xy
		xmin = numpy.min(corners[:,0])
		xmax = numpy.max(corners[:,0])
		ymin = numpy.min(corners[:,1])
		ymax = numpy.max(corners[:,1])

		frame = [[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]]
		frame = numpy.asarray(frame)

		# corners of the frame that contains the rotated image (physical space)
		self.frame_corners = frame





class Scanner(object):


	def __init__(self, sample=None):


		self.sample = sample
		
		# these are the setpoint x/y
		self.xset = 0
		self.yset = 0

	

	## only does fwd/up trace
	def ScanImage(self, pixels, size, angle):


		topo = numpy.zeros((pixels,pixels))

		dr = size / pixels

		# compute the fast/slow scan directions
		theta = angle * numpy.pi / 180.0
		vf = dr*numpy.asarray([numpy.cos(theta), numpy.sin(theta)])
		vs = dr*numpy.asarray([-numpy.sin(theta), numpy.cos(theta)])
		p0 = numpy.asarray([self.xset, self.yset])
		print("scan start at",p0)

		for i in range(pixels): # loop over px in the slowscan direction
			for j in range(pixels): # loop over px in the fastscan direction

				p = p0 + vs*i + vf*j

				topo[i,j] = self.sample.Topography(p[0],p[1])
				

		topo += (2*numpy.random.rand(topo.shape[0], topo.shape[1])-1)*0.002
		spm = SPM(topo, size, size, self.yset, self.xset, angle)
		
		# the scanner is now at the corner of the image
		self.xset += pixels*(vs[0] + vf[0])
		self.yset += pixels*(vs[1] + vf[1])

		return spm



	## target is the physical space position where we want to go.
	def MoveTip(self, target):

		self.xset = target[0]
		self.yset = target[1]


	def GetTip(self):

		return numpy.asarray([self.xset, self.yset])


if __name__ == "__main__":

	s = Sample(1000, 2.0, 0.2, 1.8, 8)
	#s = Sample.Load('sample.bin')


	topo = numpy.zeros((256,256))

	for i in range(topo.shape[0]):
		x = i*s.size / topo.shape[0]
		for j in range(topo.shape[1]):
			y = j*s.size / topo.shape[1]

			topo[i,j] = s.Topography(x,y)

			
	s.Save('sample.bin')

	plt.matshow(topo)
	plt.show()


	topo2 = numpy.zeros((256,256))

	for i in range(topo2.shape[0]):
		x = i*s.size*0.01 / topo2.shape[0]
		for j in range(topo2.shape[1]):
			y = j*s.size*0.01 / topo2.shape[1]

			topo2[i,j] = s.Topography(x,y)

	plt.matshow(topo2)
	plt.show()

