import re
import numpy as np
import numpy
import pyclipper



class GDS:


	def __init__(self, file):

		"""
		Open and parse a GDS file.
		
		:param file: GDS file to parse
		:type file: file object or string (filename)
		
		:return: GDS object interface.
		:rtype: :class:`GDSConverter.GDS`
		"""

		# list of lines in the file - list of strings - why are we keeping this?
		self.file_strings = []

		if isinstance(file,str):
			with open(file, "r") as fin:
				self.file_strings = fin.readlines()
		else:
			self.file_strings = file.readlines()

		# parse the strings and get shapes

		# dictionary of Shape objects
		self.shapes = {}
		self._to_shapes()

		print("GDS file parsed")


	def _to_shapes(self):

		"""
		Parse the file lines and compile a dictionary of shapes.
		
		"""

		# clear the dictionary
		self.shapes = {}

		# takes in the list of strings and returns a dictionary of the shapes. Each shape is given by its own
		# dictionary that contains which layer it is in and its coordinates (as an array, with the 0th column being 
		# the x_coordinates and the 1st column being the y coordinates)
		indices = [i for i, x in enumerate(self.file_strings) if x.startswith("BOUNDARY ")] # tell me the indices of 'Boundary' in the list.

		# The actual coordinates of the shape then start 3 items later and finish when we come to an 'ENDEL' item
		
		shapeID = 1
		for i in indices:

			layerID = int(self.file_strings[i+1][-3])
			n=i+3
			
			coords = []
			while not self.file_strings[n].startswith('ENDEL '):
				x = float(re.search('-?[0-9]+', self.file_strings[n])[0])
				y = float(re.search('[:]\s-?[0-9]+', self.file_strings[n])[0][2:])
				coords.append([x,y])
				n += 1
			
			vertexes = np.asarray(coords)
			shp = Shape(vertexes, shapeID, layerID)
			self.shapes[shapeID] = shp

			shapeID += 1

		


class Shape:


	def __init__(self, coordinates, index, layer):

			self.index = index
			self.layer = layer
			self.vertexes = coordinates

			# the last point should be the same as the first one
			self.normals = np.zeros(self.vertexes.shape)

			for i in range(self.vertexes.shape[0]-1):

				v = self.vertexes[i]
				prv = i - 1
				if i == 0: prv = self.vertexes.shape[0]-2
				nxt = i + 1

				vp = self.vertexes[prv]-v
				vn = self.vertexes[nxt]-v

				nm = np.linalg.norm(vp)
				nn = np.linalg.norm(vn)

				vp /= nm
				vn /= nn

				self.normals[i] = (vp+vn)*0.5
				nn = np.linalg.norm(self.normals[i])
				if nn == 0:
					raise ValueError("Shape segments must not be aligned!")
				self.normals[i] /= nn



			self.rasterPath = None
			

			self.Save()
			

	def __str__(self):

		return "shape[{}/{}]vertexes: {}".format(self.layer,self.index,self.vertexes)
	


	def pointIsInside(self, p):

		# we need to check how many times a line from p (in any direction)
		# intersects the segments of the shape

		# p + l*d == x0 + m*r
		# x0 = start of a segment
		# d any direction
		# r direction of the segment - or the segment vector not normalised even
		
		A = numpy.zeros((2,2),dtype=numpy.float64)
		A[0,0] = 1
		Ai = numpy.zeros((2,2), dtype=numpy.float64)
		crossings = 0

		for i in range(self.vertexes.shape[0]-1):

			x0 = self.vertexes[i]
			b = x0 - p
			r = self.vertexes[i+1] - x0
			A[:,1] = -r

			det = numpy.linalg.det(A)
			if det == 0: continue

			Ai = -A
			Ai[0,0] = A[1,1]
			Ai[1,1] = A[0,0]
			Ai /= det

			l,m = numpy.dot(Ai, b)
			if l < 0: continue
			if m < 0 or m > 1:
				continue
			else:
				crossings += 1
				#print("crossing at",x0+m*r,m)

		#print("total crossings",crossings)
		return (crossings % 2 != 0)



	def Save(self):

		filename = "shape-{}-{}.txt".format(self.layer,self.index)
		fout = open(filename, "w")
		for i in range(self.vertexes.shape[0]-1):
			fout.write("{} {} {} {}\n".format(self.vertexes[i,0], self.vertexes[i,1], self.normals[i,0], self.normals[i,1]))

		fout.close()


	def vector_scan(self, write_type, scan_type, pitch):

		if write_type == 'X-serpentine':
			self.rasterPath = self.x_serp(pitch)

		if write_type == 'Y-serpentine':
			self.rasterPath = self.y_serp(self.vertexes, pitch)

		if scan_type == 'Fill and outline':
			self.rasterPath = np.vstack((self.vertexes, self.rasterPath[::-1,:]))


	def x_serp(self, pitch):

		rot90 = np.array([[0,-1],[1,0]])

		# rotate the coordinates by 90 degrees
		coordsrot = np.matmul(rot90, self.vertexes.T).T

		# put this into y_serp
		full = self.y_serp(coordsrot, pitch)

		# rotate back 
		full = np.matmul(-rot90, full.T).T

		return full


	def y_serp(self, coords, pitch):

		# takes in coordinates of a shape and returns the vector coordinates needed to make a y serpentine across it
		edge_top, edge_bottom = self.y_split_up(coords) # define the top and bottom edge
		
		top_points = self.y_points_on_edge(edge_top, pitch) # find points along top spaced by pitch
		bottom_points = self.y_points_on_edge(edge_bottom, pitch) # points along bottom
		# to get the final path we need toalternate between the top and bottom points.
		# we define an array of right shape and fill it in with the right points
		full_points = np.zeros((top_points.shape[0]+bottom_points.shape[0],2))
		full_points[::2]=bottom_points
		full_points[1::2]=top_points
		# we don't have a serpentine right now but a zigzag. Need to swap some of the points around
		f = full_points[3::4,:].copy()
		full_points[3::4,:]=full_points[2::4,:].copy()
		full_points[2::4,:]=f
		return full_points

	def y_points_on_edge(self, edge, pitch):

		vert1 = edge[0,:]
		vert2 = edge[1,:]
		# x points should be evenly spaced from one side to the other
		x_points = np.linspace(edge[0,0],edge[-1,0], int((edge[-1,0]-edge[0,0]))//pitch, endpoint=True)
		# define the first set of points between first two vertices. We do it
		# outside the for loop so then we can just np.vstack to add the others on
		less_than = x_points<=vert2[0]
		more_than = vert1[0]<x_points
		x = x_points[more_than&less_than] # x points between first two vertices
		# find equation of straight line between first two vertices
		if (vert2[0]-vert1[0]==0):
			m=0 # this is in case it's a horizontal line. The array will be empty but we can't divide by 0 anyway
		else:
			m = (vert2[1]-vert1[1])/(vert2[0]-vert1[0])
		c = vert1[1]-m*vert1[0]
		y = m*x+c
		points = np.vstack((x,y)).T
		for i in range(edge.shape[0]-2):
			vert1 = edge[i+1,:]
			vert2 = edge[i+2,:]
			less_than = x_points<=vert2[0]
			more_than = vert1[0]<x_points
			x = x_points[more_than&less_than] # x points between vertices
			# find equation of straight line between vertices
			if (vert1[0]-vert2[0]==0):
				m=0 # this is in case it's a vertical line. The array will be empty but we can't divide by 0 anyway
			else:
				m = (vert2[1]-vert1[1])/(vert2[0]-vert1[0])
			c = vert1[1]-m*vert1[0]
			y = m*x+c
			new_points = np.vstack((x,y)).T
			points = np.vstack((points,new_points))
		return points


	def y_split_up(self, coords):

		# takes in coordinates of the shape and returns the top edge and bottom edge
		xmax = np.amax(coords[:,0], axis = 0)
		xmin = np.amin(coords[:,0], axis = 0) 
		top_start = [xmin, np.amax(coords[np.where(coords[:,0]==xmin),1])] # define the top left vertex (i.e. the start of top edge)
		top_end = [xmax, np.amax(coords[np.where(coords[:,0]==xmax),1])] #define top right vertex (i.e. end of top edge)
		# we take the top_start, top_end and all points inbetween as top edge. Remaining stuff is the bottom edge.
		
		# find row in coords where top_start and top_end are
		ts_index = np.where(np.sum(coords==top_start,axis=1)==2)[0][0]
		te_index = np.where(np.sum(coords==top_end,axis=1)==2)[0][0]
		# define the top edge
		if te_index<ts_index:
			# we reverse it if ts is at the end as that's the convention we work with
			top_edge = coords[te_index:ts_index+1,:][::-1,:]
			# bottom edge is whatever is left over
			bottom_edge = np.vstack((coords[:te_index+1,:], coords[ts_index:,:]))[::-1,:]
		if ts_index<te_index:
			top_edge = coords[ts_index:te_index+1,:]
			# bottom edge is whatever is left over
			bottom_edge = np.vstack((coords[:ts_index+1,:], coords[te_index:,:]))[::-1,:]
		
		# next reshuffle so bottom edge has same start point as top edge
		bottom_start = int(np.where(np.sum(bottom_edge==top_start, axis=1)==2)[0])
		bottom_edge = np.vstack((bottom_edge[bottom_start:,:], bottom_edge[:bottom_start,:]))
		
		return top_edge, bottom_edge



	def OffsetFilling(self, step):




		pass




if __name__ == "__main__":

	example = GDS('4-contact-STM.txt')

	import pyclipper
	import matplotlib.pyplot as plt

	shape = example.shapes[1]

	subj = []	
	for i in range(shape.vertexes.shape[0]-1):
		p = (shape.vertexes[i,0], shape.vertexes[i,1])
		subj.append(shape.vertexes[i])

	#subj = ((180, 200), (260, 200), (260, 150), (180, 150))

	path = subj
	polys = [path]

	plt.figure()
	coord = list(path)
	coord.append(coord[0])
	xs, ys = zip(*coord)
	plt.plot(xs,ys)

	offset = -20
	for rep in range(40):

		pco = pyclipper.PyclipperOffset()
		pco.AddPath(path, pyclipper.JT_SQUARE, pyclipper.ET_CLOSEDPOLYGON)

		solution = pco.Execute(offset)
		if len(solution) == 0: break
		solution = np.asarray(solution[0])

		path = solution
		polys.append(path)

		coord = list(path)
		coord.append(coord[0])
		xs, ys = zip(*coord)
		plt.plot(xs,ys)



	# solution (a list of paths): [[[253, 193], [187, 193], [187, 157], [253, 157]]]

	plt.show() # if you need...

	print(shape.vertexes)
	p = numpy.asarray([0,0])
	p = shape.vertexes[0]+[1,1]
	print(p,shape.pointIsInside(p))
