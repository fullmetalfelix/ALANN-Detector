import numpy
import scipy
import scipy.signal
from scipy import ndimage

import matplotlib.pyplot as plt
import math


## Returns a gaussian kernel with norm 1.
# The matrix size will be (2size+1) x (2size+1)
#
def GaussianMatrix(size, sigma):

	mat = numpy.zeros((2*size+1,2*size+1))
	acc = 0

	for i in range(-size,size+1):
		for j in range(-size,size+1):

			v = numpy.exp(-(i*i+j*j)/(2*sigma*sigma))
			mat[i+size,j+size] = v
			acc += v

	mat /= acc

	return mat


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



## Detector Object
#
#
class Detector(object):


	## Initialise a new Detector
	#
	def __init__(self):


		# parameters for the image cleaner
		# this assumes the input is a topography in [nm]
		self.spikeThreshold = 0.04
		self.offsetThreshold = 0.06
		self.tearHeightThreshold = 0.02
		self.tearLengthThreshold = 10 # in px
		self.tearGlue = 1 # px
		
		# parameters for the step detector
		self.margin = 10 # px
		self.sizeMin = 150
		self.eccMax = 0.85


		# parameter for pixel to real length conversion
		self.pixel_to_real = 5e-6/512  # these come from the image in test.py. 512 pixels and 5um.




	# ### ADVANCED CLEANUP SYSTEM ### #########################################

	## Clean up a raw image as best as we possibly can.
	# Returns a data dictionary with many intermediate products.
	#
	def ImageCleanup(self, spm):

		if not isinstance(spm,SPM):
			raise ValueError('argument is not an SPM object')
			return None

		src = spm.data

		proc2D = self.AntiSlope2D(src)
		#plt.matshow(proc2D)
		#plt.colorbar()
		#plt.savefig('proc2D.png')
		# this works ok


		despiked = self.DespikerOne(proc2D)
		#plt.matshow(despiked["output"])
		#plt.colorbar()
		#plt.savefig('despiked.output.png')

		#plt.matshow(despiked["spikemask"])
		#plt.savefig('despiked.spikemask.png')
		# this is also ok

		#print("spikemask sum:",numpy.sum(despiked["spikemask"].flatten()))
		#print("despiked sum:",numpy.sum(despiked["output"].flatten()))


		salign = self.ScanAlign(despiked['output'])
		#plt.matshow(salign["output"])
		#plt.colorbar()
		#plt.savefig('salign.output.png')

		#print(salign['lineOffsets'])
		#print("salign sum:",numpy.sum(salign["output"].flatten()))
		# this is also ok


		tears = self.TearMask(salign["output"])
		#plt.matshow(tears["mask"])
		#plt.savefig('tears.mask.png')
		
		#plt.matshow(tears["tearOffset"])
		#plt.colorbar()
		#plt.savefig('tears.tearOffset.png')

		print("tearmask sum:",numpy.sum(tears["mask"].flatten()))


		rough = salign["output"] - tears["tearOffset"];
		#plt.matshow(rough)
		#plt.colorbar()
		#plt.savefig('rough.png')

		creepinfo = self.Creeper(rough)
		#print(creepinfo)

		# now we can reconstruct the image
		data = {}
		data['input'] = proc2D

		data["spikes"] = despiked["spikes"]
		data["spikemask"] = despiked["spikemask"]
		data["lineOffsets"] = salign["lineOffsets"]
		data["tearOffset"] = tears["tearOffset"]
		data["creepInfo"] = creepinfo

		data["rough"] = rough
		

		self.RebuilderStage1(data)
		#plt.matshow(data["output"])
		#plt.colorbar()
		#plt.savefig('data.build1.png')

		self.RebuilderStage2(data)
		#plt.matshow(data["output"])
		#plt.colorbar()
		#plt.savefig('data.build2.png')

		self.RebuilderStage3(data)
		#plt.matshow(data["output"])
		#plt.colorbar()
		#plt.savefig('data.build3.png')

		self.RebuilderStage4(data)
		#plt.matshow(data["output"])
		#plt.colorbar()
		#plt.savefig('data.build4.png')

		return data


	## Returns the src image without the fitting plane slope
	# todo optimise the plane subtraction
	def AntiSlope2D(self, src):

		nrows = src.shape[0]
		ncols = src.shape[1]

		# do a plane fitting
		A = []
		b = []
		for i in range(nrows):
			for j in range(ncols):
				A.append([i,j,1.0])
				b.append(src[i,j])

		A = numpy.matrix(A)
		b = numpy.matrix(b).T

		fit = (A.T * A).I * A.T * b
		errors = b - A * fit
		residual = numpy.linalg.norm(errors)	

		print("fitting plane: %f x + %f y + %f = z" % (fit[0], fit[1], fit[2]))

		result = numpy.zeros(src.shape)

		for i in range(nrows):
			for j in range(ncols):
				result[i,j] = src[i,j] - (fit[0]*i + fit[1]*j + fit[2])
		print("desloping done")

		return result

	def DespikerOne(self, proc2D):

		ethr = self.spikeThreshold

		nrows = proc2D.shape[0]

		spkmask = proc2D*0
		proc = proc2D*0
		spikes = []

		for li in range(nrows): # loop over scan lines

			line0 = proc2D[li]

			# find all possible spike peaks where at least one side spikes more than thr
			events = []

			for ii in range(1, len(line0)-1):
				deltan = line0[ii] - line0[ii - 1]
				deltap = line0[ii] - line0[ii + 1]
				condition = (deltan > 0) and (deltap > 0)
				condition = condition and ((deltan > ethr/2) or (deltap > ethr/2));
				if condition: events.append([0, ii, 0])
			# end for

			# calculate the overall left and right fall for each event
			for i in range(len(events)):

				i0 = events[i][1]
				for j in range(i0 + 1, len(line0)): # right side
					delta = line0[i0] - line0[j]
					if delta <= 0:
						events[i][2] = i0
						break

					if j==len(line0)-1:
						events[i][2] = len(line0)-1
						break
					i0+=1
				# end of for

				i0 = events[i][1];
				for j in range(i0-1, -1, -1): # left side
					delta = line0[i0] - line0[j]
					if delta <= 0:
						events[i][0] = i0
						break
					if j==0:
						events[i][0] = 0
						break
					i0-=1
			
			
			# filter events where both left and right falls are above threshold
			deltap = [line0[e[1]] - line0[e[0]] for e in events]
			deltan = [line0[e[1]] - line0[e[2]] for e in events]
			deltas = [ [deltap[i], events[i], deltan[i]] for i in range(len(events))]
			lineevents = [ events[i] for i in range(len(events)) if deltap[i] > ethr and deltan[i] > ethr ]

			#deltas = Select[deltas, #[[1]] > ethr && #[[3]] > ethr &];
			#lineevents = events = deltas[[All, 2]];
			spikes.append(lineevents)
			

			# interpolate in the event regions
			lineOut = line0;
			for e in lineevents:
				i0 = e[0]
				i1 = e[2]
				for i in range(i0+1, i1-1+1):
					t = (i - i0)/(i1 - i0)
					lineOut[i] = line0[i0]*(1 - t) + line0[i1]*t
					spkmask[li, i] = 1
				# end of loop over points {i, i0 + 1, i1 - 1}];
			# end of loop over events
			proc[li] = lineOut
			#print(li, numpy.sum(spkmask[li]))
		# end of loop over lines , {li, Length[proc2D]}];
		
		return {'spikemask': spkmask, 'output': proc, 'spikes': spikes}

	def DespikerOneLiner(self, proc2D, ethr):

		spkmask = proc2D*0
		nrows = proc2D.shape[0]

		for li in range(nrows): # loop over scan lines

			line0 = proc2D[li] 
			deltas = numpy.abs(line0[1:] - line0[0:-1])

			# find all possible spike peaks where at least one side spikes more than thr
			events = []
			for ii in range(1, line0.shape[0]-1):
				deltan = line0[ii] - line0[ii - 1];
				deltap = line0[ii] - line0[ii + 1];
				condition = (deltan > 0) and (deltap > 0)
				condition = condition and ((deltan > ethr) and (deltap > ethr));
				if condition: events.append([0,ii,0])
			

			# set mask in the event regions
			for e in events: spkmask[li, e[1]] = 1
			
		# end of loop

		return {'mask': spkmask, 'events': events}

	def ScanAlign(self, image):

		tmp = image*1
		nrows = image.shape[0]
		ncols = image.shape[1]
		thr = self.offsetThreshold


		lineOffsets = [0 for i in range(nrows)]

		for i in range(1, nrows):
			
			delta = tmp[i] - tmp[i - 1]
			idx = [n for n in range(ncols) if numpy.abs(delta[n]) < thr]
			offset = 0

			if len(idx) < 0.5*nrows:
				#print('if true for ',i)
				displ = [0,0]
				stp = 0.002
				for l in numpy.arange(-0.16, 0.16+stp, stp):
					di = [n for n in range(ncols) if numpy.abs(delta[n] + l) < thr]
					if len(di) > displ[1]:
						displ = [l, len(di)]

				displ = displ[0] # get the offset
				delta += displ
				idx = [n for n in range(ncols) if numpy.abs(delta[n]) < thr]
				offset -= displ
			
			if len(idx) == 0: continue
			
			offset += numpy.mean(delta[idx])
			if numpy.abs(offset) > 0.005:
				tmp[i:] -= offset
				lineOffsets[i:] -= offset
			
		# end of loop over rows

		return {"output": tmp, "lineOffsets": lineOffsets}
		
	def TearMask(self, src):

		nrows = src.shape[0]

		despikeV = self.DespikerOneLiner(numpy.transpose(src), self.tearHeightThreshold)
		spkmaskV = despikeV['mask']
		tmp = despikeV['events']

		#print("despikeV sum:",numpy.sum(spkmaskV.flatten()))

		spkmaskV = numpy.transpose(spkmaskV).astype(numpy.int32)
		mask = src*0
		maskOffset = src*0


		for li in range(0, nrows):

			ml = spkmaskV[li]
			
			# remove single pixel spikes *)
			for ig in range(1, self.tearGlue+1):
				for j in range(ml.shape[0]-ig):
					
					seg = ml[j : j + ig+1]
					if seg[0] != seg[-1]: continue
					if seg[0] == seg[1]: continue

					unique = Set(seg[1:-1])
					if len(unique) != 1: continue
					
					#print("tagged",li,j,j + ig + 1,seg[0])
					ml[j : j + ig + 1] = seg[0]

			
			# find segments that are long enough
			segments = []; s = 0; seg = [0, 0];
			for i in range(0, ml.shape[0]):
				if ml[i] == 1 and s == 0:
					s = 1
					seg[0] = i
					#print("on",li,i)
					continue

				if ml[i] == 0 and s == 1:
					s = 0
					seg[1] = i - 1
					segments.append([seg[0],seg[1]])
					#print("off",li,i-1)
					continue
			
			#if li == 70: print(segments)

			#tmpsegs = []
			#for s in segments:
			#	if s[1]-s[0] > self.tearLengthThreshold:
			#		tmpsegs.append(s)
			#segments = tmpsegs
			segments = [s for s in segments if s[1]-s[0] > self.tearLengthThreshold]
			#if len(segments) > 0: print(li,len(segments))

			if li > 0 and li < nrows-1:
				for s in segments:
					prevl = src[li - 1, s[0] : s[1]+1]
					nextl = src[li + 1, s[0] : s[1]+1]
					nextl = 0.5*(prevl + nextl)
					prevl = src[li, s[0] : s[1]+1] - nextl
					prevl = numpy.mean(prevl)
					mask[li, s[0] : s[1]+1] = 1
					maskOffset[li, s[0] : s[1]+1] = prevl
				
			
		# end of loop over lines
		return {'mask': mask, 'tearOffset': maskOffset}

	def Creeper(self, rough):

		nrows = rough.shape[0]
		thr = self.offsetThreshold

		creeplines = [] # check for piezo creep lines
		for i in range(0, nrows):
			
			deltas = [1, 1]
			if i != 1: deltas[0] = numpy.mean(numpy.abs(rough[i] - rough[i - 1]))
			if i != nrows-1: deltas[1] = numpy.mean(numpy.abs(rough[i] - rough[i + 1]))
			if (deltas[0] > thr) and (deltas[1] > thr): creeplines.append(i)


		# creep lines should be contiguous!
		adjacent = True
		edge = True

		if len(creeplines) > 0:
			
			for i in range(1, len(creeplines)):
				adjacent = adjacent and (creeplines[i] == creeplines[i - 1] + 1)
			
			edge = (creeplines[0] == 0) or (creeplines[-1] == nrows-1)
		
		if not adjacent: print("Creep lines are not adjacent!")
		if not edge: print("Creep lines are not at the edge of the image!")

		return [creeplines, adjacent, edge]



	def RebuilderStage1(self, data):

		rebuilt = data["input"]
		ci = data["creepInfo"]

		nrows = rebuilt.shape[0]

		# realign the scanlines using the estimated offsets 
		for i in range(nrows): rebuilt[i] += data["lineOffsets"][i]
		rebuilt -= data["tearOffset"]; # recover from tip changes
		data['output'] = rebuilt


		if ci[1] and ci[2]: # cood creepinfo

			idx = [i for i in range(nrows) if not (i in ci[0])]

			data["input"] = data["input"][idx]
			data["rough"] = data["rough"][idx]
			data["output"] = data["output"][idx]
			data["spikemask"] = data["spikemask"][idx]
			data["spikes"] = [data['spikes'][i] for i in idx]

			if len(ci[0]) > 0:
				if ci[0][0] == 0: 
					data["x0"] = [ci[0][0], 0]
		
	def RebuilderStage2(self, data):
		
		recmask = data["spikemask"]
		spikes = data["spikes"]
		rebuilt = data["output"]

		nrows = rebuilt.shape[0]

		# spikes in the first/last line are recovered with simple linear interpolation along the line
		for li in range(nrows):
			lineOut = rebuilt[li];
			
			for e in spikes[li]: 
				i0 = e[0]; i1 = e[2];
			
				for i in range(i0+1, i1-1+1):
					t = (i - i0)/(i1 - i0)
					lineOut[i] = lineOut[i0]*(1 - t) + lineOut[i1]*t
			

			rebuilt[li] = lineOut;
		
		recmask[0] *= 0;
		recmask[-1] *= 0;

		# spikes in the first/last column are recovered with simple linear interpolation
		# there cannot be any!

		data["recmask"] = recmask
		data["output"] = rebuilt
		
	def RebuilderStage3(self, data):
		
		recmask = data["recmask"]
		rebuilt = data["output"]

		nrows = rebuilt.shape[0]
		ncols = rebuilt.shape[1]

		fixed = 1

		while fixed != 0:
			
			# all isolated spikes can be interpolated using neighbour lines

			fixed = 0
			for i in range(1,nrows-1):
				for j in range(ncols):

					if (recmask[i - 1, j] == 0) and (recmask[i + 1, j] == 0) and (recmask[i, j] == 1):
						rebuilt[i, j] = 0.5*(rebuilt[i - 1, j] + rebuilt[i + 1, j])
						recmask[i, j] = 0
						fixed += 1
			
			fixed = 0
			
			# now we can process single points along scanlines
			for i in range(nrows):
				for j in range(1,ncols-1):
					if (recmask[i, j - 1] == 0) and (recmask[i, j + 1] == 0) and (recmask[i, j] == 1):
						rebuilt[i, j] = 0.5*(rebuilt[i, j - 1] + rebuilt[i, j + 1])
						recmask[i, j] = 0
						fixed += 1
		
		# end of while loop

		data["recmask"] = recmask
		data["output"] = rebuilt

	def RebuilderStage4(self, data): 

		recmask = data["recmask"]
		rebuilt = data["output"]

		nrows = rebuilt.shape[0]
		ncols = rebuilt.shape[1]


		for i in range(1,nrows-1):
			for j in range(1, ncols-1):

				if recmask[i, j] == 0: continue

				interp = [[False, False], [False, False]]

				j1 = -1
				for j1 in range(j + 1, ncols):
					if recmask[i, j1] == 0: 
						interp[0][1] = True
						break
				
				j0 = -1
				for j0 in range(j - 1, -1, -1):
					if recmask[i, j0] == 0:
						interp[0][0] = True
						break
				
				i1 = -1
				for i1 in range(i + 1, nrows):
					if recmask[i1, j] == 0:
						interp[1][1] = True
						break
				
				i0 = -1
				for i0 in range(i - 1, -1, -1):
					if recmask[i0, j] == 0:
						interp[1][0] = True
						break

				deltai = i1 - i0
				deltaj = j1 - j0
				interp[0] = interp[0][0] and interp[0][1]
				interp[1] = interp[1][0] and interp[1][1]

				if interp[0] and (deltai < deltaj):
					# interpolate along line
					for jj in range(j0+1, j1-1+1):
						t = (jj - j0)/(j1 - j0)
						rebuilt[i, jj] = rebuilt[i, j0]*(1 - t) + rebuilt[i, j1]*t
						recmask[i, jj] = 0

					continue

				if interp[1] and (deltaj <= deltai):
					# interpolate across lines
					for ii in range(i0+1, i1-1+1):
						t = (ii - i0)/(i1 - i0);
						rebuilt[ii, j] = rebuilt[i0, j]*(1 - t) + rebuilt[i1, j]*t
						recmask[ii, j] = 0

					continue


		data["recmask"] = recmask
		data["output"] = rebuilt

	# #########################################################################
	# ### MEDIUM SCALE IMAGES #################################################


	def Mesoscale_line_peakfind(line, stepthr=0.07, stepminheight=0.1):


		state = 0
		slst = 0
		delta = 0
		steps = []
		peakheads = []

		for i in range(1, len(line)): # loop over pixels

			if state == 0:

				slst = i-1
				if line[i] > line[i-1]:
					state = 1
				else:
					state = -1

				continue

			if state == 1 and line[i]<line[i-1]: # trend was going up and it stopped

				# did it go up enough to make a step?
				delta = line[i-1]-line[slst]

				if numpy.abs(delta) > stepthr:
					steps.append([slst, i-1, delta, int(numpy.round(delta/stepminheight))])

				peakheads.append([i-1,1])
				state = 0
			
			if state==-1 and line[i]>line[i-1]: # trend was going down and it stopped

				# did it go down enough to make a step
				delta = line[i-1] - line[slst]
				
				if numpy.abs(delta) > stepthr:

					steps.append([slst, i-1, delta, int(numpy.round(delta/stepminheight))])

				peakheads.append([i-1,-1])
				state = 0
		# end of loop over pixels

		if peakheads[0][0] != 1:
			peakheads.insert(0, [0, -peakheads[0][1]])

		if peakheads[-1][0] != len(line):
			peakheads.append([len(line)-1, -peakheads[-1][1]])

		#steps = numpy.asarray(steps)
		peakheads = numpy.asarray(peakheads)
		
		#print(steps)
		#print(peakheads)

		peaks = []
		for i in range(1,peakheads.shape[0]-1):
			
			if numpy.abs(line[peakheads[i, 0]] - line[peakheads[i + 1, 0]]) > 0.04: peaks.append((peakheads[i-1,0],peakheads[i,0],peakheads[i+1,0]))
			if numpy.abs(line[peakheads[i, 0]] - line[peakheads[i - 1, 0]]) > 0.04: peaks.append((peakheads[i-1,0],peakheads[i,0],peakheads[i+1,0]))
			
		peaks = list(dict.fromkeys(peaks)) # delete duplicates
		#print(peaks)

		return peaks, steps


	def Mesoscale_line_terracedivide(line, peaks, steps):

		start = 0

		tmp = [[i,line[i]] for i in range(len(line))]
		tmp = numpy.asarray(tmp)
		mask = numpy.zeros(len(line)) + 1

		# also take out the spikes first
		for b in peaks: mask[b[0] : b[2]] = -1

		terraces = []

		# split terraces
		for s in steps:

			tmp2 = tmp[start : s[0]]
			tmp3 = tmp2[mask[start : s[0]] > 0]

			start = s[1]
			
			terraces.append(numpy.asarray(tmp3))
		
		if  start < len(line):
			tmp2 = tmp[start:]
			tmp3 = tmp2[mask[start:] > 0]
			terraces.append(numpy.asarray(tmp3))
		
		tmp = [x for x in terraces if len(x) > 0]
		terraces = tmp
		#terraces = numpy.asarray(terraces)
		

		return terraces
		
	
	def Mesoscale_line_goodsteps(line, terraces, steps, stepthr=0.09, stepPhysSize=0.12):

		tstarts = [t[0,0] for t in terraces]
		tends = [t[-1,0] for t in terraces]
		goodsteps = numpy.zeros(len(steps))

		for i in range(len(steps)):

			# find the terrace before this step
			t1 = [t for t in terraces if t[-1,0] <= steps[i][0]]
			if len(t1) == 0:
				continue
			else:
				t1 = t1[-1]

			# find the terrace after this step
			t2 = [t for t in terraces if t[0,0] >= steps[i][1]]
			if len(t2) == 0: continue
			else: t2 = t2[0]

			# check delta of means
			delta = numpy.mean(t2[:,1]) - numpy.mean(t1[:,1])
			
			# if delta is below threshold, then there must have been a false step edge
			if numpy.abs(delta) > stepthr:
				goodsteps[i] = numpy.round(delta/stepPhysSize)
		
		#print(goodsteps)

		tmp = []
		for i in range(len(steps)):
			if goodsteps[i] == 0: continue
			s = steps[i]
			#print(s)
			s[-1] = goodsteps[i]
			tmp.append(s)

		return tmp


	def Mesoscale_line_terraceRedivide(line, goodsteps, terraces):

		mterraces = []

		# merge terraces before a step
		tidx = 0
		sidx = 0
		cend = 0
		cterrace = None

		while True:

			if tidx == len(terraces):
				mterraces.append(cterrace)
				break

			t = terraces[tidx]

			# if we past the last step already
			if sidx >= len(goodsteps):
				cterrace = numpy.concatenate((cterrace, t))
				tidx += 1
				continue

			# check if terrace tidx ends before the current step
			if t[-1,0] <= goodsteps[sidx][0]:
				
				if cterrace is None: cterrace = t
				else: cterrace = numpy.concatenate((cterrace, t))

				tidx += 1 # move to next terrace
				continue

			# if the current terrace starts past the current step
			# save the merged terrace and move to the next step
			if t[0,0] >= goodsteps[sidx][1]:

				mterraces.append(cterrace)
				cterrace = t
				sidx += 1
				tidx += 1


		# compute linear fits and subtract
		for i in range(len(mterraces)):

			t = mterraces[i]
			x = t[:,0]
			y = t[:,1]

			if t.shape[0] < 5: continue
			
			#print("polyfit",t.shape)
			fit = numpy.polyfit(x,y,1)
			
			mterraces[i][:,1] -= fit[1] + fit[0] * x


		# assign terrace height/ID values
		lineID = numpy.zeros(len(line), dtype=numpy.int32)
		stepmask = numpy.zeros(len(line), dtype=numpy.int32)
		for g in goodsteps: 
			lineID[g[1]] = g[-1]
			stepmask[g[0]:g[1]] = 1

		lineID = numpy.cumsum(lineID)

		return mterraces, lineID, stepmask


	def Mesoscale_image_process(stmscan, st1=0.07, sm1=0.1, st2=0.09, sps=0.12):

		img = stmscan.data
		stepmask = numpy.zeros(img.shape, dtype=numpy.int32)
		terraceIDs = numpy.zeros(img.shape, dtype=numpy.int32)


		for li in range(img.shape[0]):
			line = stmscan.data[li]
			peaks, steps = Detector.Mesoscale_line_peakfind(line, stepthr=st1, stepminheight=sm1)
			linemask = Detector.Mesoscale_line_bumpfind(line, steps, thr=0.07, buffersize=8)
			stepmask[li] = linemask

			''' FIRST TEST
				peaks, steps = Detector.Mesoscale_line_peakfind(line, stepthr=st1, stepminheight=sm1)
				terraces = Detector.Mesoscale_line_terracedivide(line, peaks, steps)
				goodsteps = Detector.Mesoscale_line_goodsteps(line, terraces, steps, stepthr=st2, stepPhysSize=sps)
				mterraces, lineID, linemask = Detector.Mesoscale_line_terraceRedivide(line, goodsteps, terraces)

				terraceIDs[li] = lineID
				stepmask[li] = linemask
			'''


		# redo the pass in vertical direction
		for li in range(img.shape[1]):
			line = stmscan.data[:,li]
			peaks, steps = Detector.Mesoscale_line_peakfind(line, stepthr=0.08, stepminheight=0.1)
			linemask = Detector.Mesoscale_line_bumpfind(line, steps, thr=0.07, buffersize=8)
			stepmask[:,li] += linemask


		stepmask[stepmask>0] = 1


		
		#labels, nb = ndimage.label(stepmask)
		numpy.save("meso.stepmask.npy", stepmask)

		fig, (ax1, ax2) = plt.subplots(1, 2)
		ax1.matshow(stepmask)
		ax2.matshow(terraceIDs)
		plt.show()




		return terraceIDs


	def Mesoscale_image_coloring(tmap):

		# start with ...

		pass



	def Mesoscale_line_bumpfind(line, steps, thr=0.07, buffersize=4):

		mask = numpy.zeros(len(line), dtype=numpy.int32)

		for s in steps:
			
			x1 = max(0,s[0]-buffersize)
			x2 = s[1]
			x3 = min(x2+buffersize, len(line))

			b1 = line[x1:s[0]]
			b2 = line[s[1]:x3]

			if len(b1) == 0 or len(b2) == 0: continue

			m1 = numpy.mean(b1)
			m2 = numpy.mean(b2)
			delta = m2-m1

			if numpy.abs(delta) > thr:
				mask[s[0]:s[1]] = 1


		return mask


	# #########################################################################
	# #########################################################################
	# ### STEP EDGE DETECTION ### #############################################


	## Detects step edges (not 100% perfect, depending on noise, but good enough)
	# Returns a binary map (step = 1, terrace = 0)
	#
	def StepDetector(self, spm):

		if not isinstance(spm,SPM):
			raise ValueError('argument is not an SPM object')
			return None
		
		src = spm.data
		border = self.margin

		
		# TODO: make this resolution independent
		gaussmat_1_14 = GaussianMatrix(1,1.4)
		gaussmat_2_14 = GaussianMatrix(2,1.4)

		sobelkernel = numpy.asarray([
			[1/numpy.sqrt(2), 1, 1/numpy.sqrt(2)], 
			[1, -(4 + 4/numpy.sqrt(2)), 1], 
			[1/numpy.sqrt(2), 1, 1/numpy.sqrt(2)]])

		smooth = src * 1

		for smoothCycles in range(1):

			# this one has 1 px boundary of crap
			tmp0 = scipy.signal.convolve2d(smooth, gaussmat_1_14, mode='same', boundary='fill', fillvalue=0)
			#tmp0 = ListConvolve[GaussianMatrix[{1, 1.4}], smooth, {-1, 1}];
			
			# this one has 2 px of boundary crap
			tmp = scipy.signal.convolve2d(smooth, gaussmat_2_14, mode='same', boundary='fill', fillvalue=0)
			#tmp = ListConvolve[GaussianMatrix[{gaussR, 1.4}], smooth, {-1, 1}];

			smooth[1:-1, 1:-1] = tmp0[1:-1, 1:-1]
			smooth[2:-2, 2:-2] = tmp[2:-2, 2:-2]


		
		ithr = 0.02
		sobel = scipy.signal.convolve2d(smooth, sobelkernel, mode='same', boundary='fill', fillvalue=0)
		sobel = sobel[1:-1, 1:-1] # throw away borders
		sobel = numpy.abs(sobel)

		# everything less or equal ithr becomes 0
		thr = (sobel > ithr) * sobel


		
		# gaussR = 1;
		# progressive thresholding
		for cycles in range(24):

			tmp = scipy.signal.convolve2d(thr, gaussmat_1_14, mode='same', boundary='fill', fillvalue=0)
			thr[1:-1, 1:-1] = tmp[1:-1, 1:-1]
			
			thr += sobel
			thr = (thr > ithr) * thr
			
			ithr *= 1.1
		

		thr = (thr > ithr)
		thr = thr[border:-border, border:-border]
		

		# do connected component analysis

		labels, nb = scipy.ndimage.label(thr)


		# labels is a matrix with the obj IDs of connected objects
		# compute CC areas
		unique, counts = numpy.unique(labels, return_counts=True)
		cc_count = dict(zip(unique, counts))
		#print(cc_count)

		cc_count.pop(0, None) # remove bg

		# remove smalls
		keys = list(cc_count.keys())
		for l in keys:
			if l == 0: continue

			if cc_count[l] < self.sizeMin:
				labels = (labels != l) * labels
				cc_count.pop(l, None)

		
		keys = list(cc_count.keys())
		for l in keys: # for each component

			# find min and max idx of the component
			sl = scipy.ndimage.find_objects(labels == l)[0]
			
			tmp = labels[sl]
			tmp = (tmp == l)

			#plt.matshow(tmp)
			#plt.savefig('thr4_'+str(l)+'.png')

			# compute moments
			area = cc_count[l] # m00

			m11 = 0; m20=0; m02=0;
			# first the centroid
			x=0;y=0;
			for i in range(tmp.shape[0]):
				for j in range(tmp.shape[1]):
					if tmp[i,j] == 0: continue
					x += j
					y += i
			x = x/area
			y = y/area

			for i in range(tmp.shape[0]):
				for j in range(tmp.shape[1]):
					if tmp[i,j] == 0: continue

					m11 += (i-y)*(j-x)
					m02 += (j-x)*(j-x)
					m20 += (i-y)*(i-y)

			a1 = m20+m02+numpy.sqrt(math.pow(m20 - m02,2) + 4*m11*m11)
			a2 = m20+m02-numpy.sqrt(math.pow(m20 - m02,2) + 4*m11*m11)
			ecc = numpy.sqrt(1 - a2/a1)
			#ecc  = m20+m02+numpy.sqrt(math.pow(m20 - m02,2) + 4*m11*m11)
			#ecc /= m20+m02-numpy.sqrt(math.pow(m20 - m02,2) + 4*m11*m11)

			theta = 0.5*numpy.arctan2(2*m11, (m20 - m02))
			theta = numpy.abs(numpy.sin(theta))

			
			if ecc > self.eccMax and theta > 0.9:
				labels = (labels != l) * labels
				cc_count.pop(l, None)

			print(l, area, ecc, theta)


		final = (labels > 0)

		return final

	
	# #########################################################################
	# #########################################################################


# END OF CLASS
