import numpy
from detector import *
import matplotlib.pyplot as plt


# load a test image file
src = numpy.load("testimage.npy")
src *= 10.0**9 # convert topography to nm


# create a scanner with default parameters
scanner = Detector()

# clean up the image from tears and small tip changes
data = scanner.ImageCleanup(src)
cln = data['output']


# save the resulting image
plt.matshow(cln)
plt.colorbar()
plt.savefig('cleanedup.png')

# save as data too
numpy.save("cleaned.npy", cln)

# load the cleaned up image
#cln = numpy.load("cleaned.npy")

# run the step detector
stepmap = scanner.StepDetector(cln)

# save the resulting image
plt.matshow(stepmap)
plt.colorbar()
plt.savefig('stepmap.png')


