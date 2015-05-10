# 1imMosaic.py

# Mosaic builder
# by Danny Shum
# April-May, 2015
# COS 435 with Andrea LaPaugh

# This program builds a photo-mosaic out of a user-supplied image. 

# Dependencies: Numpy, PIL, SciPy, Pickle, image_slicer (by Sam Dobson)

# README:
# This program accepts 2 arguments: [filename], [number of tiles]   in that order.
# sample execution: 
# 		python 1imMosaic.py "4k_8ball.jpg" 5000
# 	will break the image "4k_8ball.jpg" into 5000 tiles.

import numpy
import PIL
from PIL import Image
from PIL import ImageStat
import struct
import scipy
import scipy.misc
import scipy.cluster
import scipy.spatial
import sys
from collections import OrderedDict
import cPickle as pickle
import pprint
import image_slicer

PREFIX = 'im'					# image prefix (in library)
SUFFIX = '.jpg'					# image type to be catalogued
DB_FILE = 'data_DB.txt'			# database output file name
RGB_FILE = 'data_RGB.txt'		# RGB list output file name

#=============================================================================#

# -- Returns the dominant color in imgFileName --
# This method is used for analyzing the library of images that will compose the 
# final mosaic.
def getDomColor( imgFileName ):
	# Reference:
	# 	http://stackoverflow.com/questions/3241929/
	# 	python-find-dominant-most-common-color-in-an-image

	# number of k-means clusters
	NUM_CLUSTERS = 4

	# Open target image
	im = Image.open(imgFileName)
	im = im.resize((150, 150))      # optional, to reduce time
	ar = scipy.misc.fromimage(im)
	shape = ar.shape
	ar = ar.reshape(scipy.product(shape[:2]), shape[2])
	ar = ar.astype(float)

	# Find clusters
	codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
	# print 'cluster centres:\n', codes

	vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
	counts, bins = scipy.histogram(vecs, len(codes))    # count occurrences

	# Find most frequent
	index_max = scipy.argmax(counts)                    
	peak = codes[index_max]
	color = ''.join(chr(int(c)) for c in peak).encode('hex')
	#print 'most frequent is %s (#%s)' % (peak, color)

	# Extra Bonus coolbeans: save image using only the N most common colors
	# c = ar.copy()
	# for i, code in enumerate(codes):
	#     c[scipy.r_[scipy.where(vecs==i)],:] = code
	# scipy.misc.imsave(imgFileName[:-4] + "CLUS" + ".jpg", c.reshape(*shape))
	# print 'saved clustered image'

	return (peak, color) 

#=============================================================================#

# -- Returns the dominant color in a PILLOW IMAGE parameter --
# This method is used for analyzing the tiles of a user-supplied image to be 
# re-created by a mosaic. The only difference from the above method is that
# this method accepts a PIL Image instead of a file name. 
def getDomIMAGEColor( imName ):
	# Reference:
	# 	http://stackoverflow.com/questions/3241929/
	# 	python-find-dominant-most-common-color-in-an-image

	# number of k-means clusters
	NUM_CLUSTERS = 4

	# Open target image
	im = imName
	im = im.resize((150, 150))      # optional, to reduce time
	ar = scipy.misc.fromimage(im)
	shape = ar.shape
	ar = ar.reshape(scipy.product(shape[:2]), shape[2])
	ar = ar.astype(float)

	# Find clusters
	codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
	vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
	counts, bins = scipy.histogram(vecs, len(codes))    # count occurrences

	# Find most frequent
	index_max = scipy.argmax(counts)                    
	peak = codes[index_max]
	color = ''.join(chr(int(c)) for c in peak).encode('hex')

	return (peak, color)

#=============================================================================#

# -- Build a dictionary of images and their stats --
# Accepts name prefix, and number of images to read
# e.g. prefix = im, suffix = jpg for im1.jpg
def buildLib( prefix, numImages ):

	#SUFFIX = '.jpg'				# image type to be catalogued
	#DB_FILE = 'data_DB.txt'		# database output file name
	#RGB_FILE = 'data_RGB.txt'		# RGB list output file name
	db = OrderedDict([])			# ordered image database
	ltRGB = []						# parallel tuple of RGB

	# read images and get dominant color. link in dictionary and store. 
	for i in range(numImages):
		imgFileName = prefix + str(i+1) + SUFFIX  
		listRGB, dominantColor = getDomColor(imgFileName)
		#RGB.append(listRGB)		# build parallel RGB lists
		tupleRGB = tuple(listRGB)
		ltRGB.append(tupleRGB)		# build parallel RGB tuples
		db[imgFileName] = { "RGB" : tupleRGB, \
							"color" : dominantColor, \
							"used" : numUses}
		print "  indexed " + imgFileName + "..."

	# Write database as pickle to file
	with open(DB_FILE, 'w') as outfile:
		print "  writing image library to " + DB_FILE + "..."
		pickle.dump(db, outfile, 2)

	# Write RGB as pickle to file
	with open(RGB_FILE, 'w') as outfile:
		print "  writing RGB library to " + RGB_FILE + "..."
		pickle.dump(ltRGB, outfile, 2)

	# build KDTree for nearest neighbor lookup
	print "  initializing kdTree..."
	kdTree = scipy.spatial.KDTree(ltRGB)
	
	return kdTree, db

#=============================================================================#
 
# -- Load image library and RGB data from local pickle files --
# So that you don't have to re-analyze 25,000 pictures every time you run
# the program. 

def loadLib( RGBFile, dbFile ):
	print '  Loading image database and RGB from files...'
	fIm = open(dbFile)
	fRGB = open(RGBFile)
	db = pickle.load(fIm)
	ltRGB = pickle.load(fRGB)

	print '  Initializing kdTree...'
	kdTree = scipy.spatial.KDTree(ltRGB)

	return kdTree, db

#=============================================================================#

# -- Make a mosaic of tim (target image) out of existing photos in the lib --
def createMosaic( tim0, numTiles0, tree0, db0 ):
	tim = tim0
	numTiles = numTiles0
	kdTree = tree0
	db = db0

	# slice tim into tiles
	print "  Slicing " + tim + " into " + str(numTiles) + " tiles..."
	tiles = image_slicer.slice(tim, numTiles, save=False)

	# Replace slices with nearest neighbors from library
	print "  Replacing tiles with nearest neighbors..."
	for tile in tiles:
		tileRGB, tileColor = getDomIMAGEColor(tile.image)
		neighborDist, neighborIndex = kdTree.query(tileRGB)
		neighborFile = PREFIX + str(neighborIndex + 1) + SUFFIX
		iNbor = Image.open(neighborFile)
		# print tile.image.size
		iNbor = iNbor.resize(tile.image.size)
		tile.image.paste(iNbor)
	
	print "  Joining new tiles..."
	# image_slicer.save_tiles(tiles, prefix=tim[:-4])
	newMosaic = image_slicer.join(tiles)

	# Save the new image under the old filename appended with "_mosaic" and 
	# the number of tiles, e.g. 'image_mosaic2000.jpg'
	newName = tim[:-4] + "_mosaic" + str(numTiles) + SUFFIX
	print "  Saving new mosaic under " + newName
	newMosaic.save(newName)

	return newMosaic

#=============================================================================#

def main():
	kdTree, db = loadLib(RGB_FILE, DB_FILE)
	im = createMosaic(sys.argv[1], int(sys.argv[2]), kdTree, db)
	im.show()
	print 'process completed successfully!'


if __name__ == "__main__":
    main()

# kdTree, db = buildLib("im", 25000)    	# Command to build the photo library out of 25000
# kdTree, db = loadLib(DB_FILE, RGB_FILE)	# Command to load the photo library data



