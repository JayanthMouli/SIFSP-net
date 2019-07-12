import numpy as np
from keras.layers import Dense, Activation
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import csv
import scipy.misc as sm
import cv2
from PIL import Image
import pandas
from glob import glob
from osgeo import gdal
import os
import rasterio as rio
import utm
import richdem as rd
from scipy.misc import toimage

raw = cv2.imread('burnedcut.jpg',0)

before_image = glob("/media/jayanthmouli/54EC-046F/before_fire_images/*")
before_image.sort()
after_image = glob("/media/jayanthmouli/54EC-046F/burned area/*")
after_image.sort()
dem = rd.LoadGDAL("/media/jayanthmouli/54EC-046F/elevation/the_better_one.tif", no_data = 0)
slope = rd.TerrainAttribute(dem, attrib='slope_riserun')
aspect = rd.TerrainAttribute(dem, attrib='aspect')

gain = {}
bias = {}
qc_lmax = {}
qc_lmin = {}
lmax = {}
lmin = {}


   
fp = open('/media/jayanthmouli/54EC-046F/AFTER_MTL.txt', 'r')
for line in fp:
	if(line.find("RADIANCE_MULT_BAND")>=0):
		s = line.split("=")
		the_band = int(s[0].split("_")[3])
		gain[the_band] = float(s[-1])
		#print float(s[-1])
	elif(line.find("RADIANCE_ADD_BAND")>=0):
		s = line.split("=")
		the_band = int(s[0].split("_")[3])
		bias[the_band] = float(s[-1])


#print after_image
band2 = gdal.Open(before_image[1])
band2data = band2.ReadAsArray().astype(np.float)
band3 = gdal.Open(before_image[2])
band3data = band3.ReadAsArray().astype(np.float)
band4 = gdal.Open(before_image[3])
band4data = band4.ReadAsArray().astype(np.float)
band5 = gdal.Open(before_image[4])
band5data = band5.ReadAsArray().astype(np.float)
band10 = gdal.Open(before_image[9])
band10data = band10.ReadAsArray().astype(np.float)
band11 = gdal.Open(before_image[10])
band11data = band11.ReadAsArray().astype(np.float)

elevation = gdal.Open("/media/jayanthmouli/54EC-046F/elevation/the_better_one.tif")
elevationdata = elevation.ReadAsArray().astype(np.float)

band3_after = gdal.Open(after_image[2])
band3_after_data = band3_after.ReadAsArray().astype(np.float)
band5_after = gdal.Open(after_image[4])
print band3_after_data.shape
band5_after_data = band5_after.ReadAsArray().astype(np.float)
band7_after = gdal.Open(after_image[6])
band7_after_data = band7_after.ReadAsArray().astype(np.float)

band1_TOAR = []
band2_TOAR = band2data
band3_TOAR = band3_after_data
band4_TOAR = band4data
band5_TOAR = band5_after_data
band6_TOAR = []
band7_TOAR = band7_after_data
band8_TOAR = []
band9_TOAR = []


xoff, a, b, yoff, d, e = band3.GetGeoTransform()
xoff_E, a_E, b_E, yoff_E, d_E, e_E = elevation.GetGeoTransform()

def pixel2coord(x, y):
 #Returns global coordinates from pixel x, y coords
 xp = a * x + b* y + xoff
 yp = d * x + e * y + yoff
 return([xp, yp])
def coord2pixel(xp, yp):
	vars = np.array([[a_E,b_E], [d_E, e_E]])
	sol = np.array([xp-xoff_E, yp-yoff_E])
	return np.linalg.solve(vars, sol)
def windangle(x1, y1, x, y):
	theta = 4
	if x==x1:
		if y1>y:
			theta = 1.35
		else:
			theta = 0.45
	elif x>x1:
		if y1>y:
			theta = 0.9
		else:
			theta = 0
	elif x<x1:
		if y1>y:
			theta = 1.8
		else:
			theta = 0.9
	elif y1==y:
		if x1>x:
			theta = 0.45
		else:
			theta = 1.35
	return theta
	
def isBurnt(x, y, mask):
	if mask[x][y] == [255]:
		return 1
	else:
		return 0

ary = []

ndvi = (band5data - band4data) / (band5data + band4data)
thiscount = 0
total = 0
burn = 0
zeroes = raw
headers = []
isBurn = 0
with open('whitepixels.csv', 'w') as csvoutput:
	writer = csv.writer(csvoutput)
	headers.append('x')
	headers.append('y')
	headers.append("NDVI")
	headers.append("TIR1")
	headers.append("TIR2")
	headers.append("elevation")
	headers.append("slope")
	headers.append("aspect")
	writer.writerow(headers)
	for ex in range(len(raw)):
		for why in range(len(raw[ex])):
			total+=1
			print 'yahey'
			X = ex+ 3826
			Y = why + 1572
			if raw[ex][why] == 0:
				isBurn = 1
				burn+=1
			else:
				isBurn = 0
			chars = []
			NDVI = ndvi[X][Y]
			tir1 = band10data[X][Y]
			tir2 = band11data[X][Y]
			UTM = pixel2coord(X,Y)
			coords = utm.to_latlon(UTM[0], UTM[1], 10, 'S')
			elev_indices = coord2pixel(coords[1], coords[0])
			elev_x = int(elev_indices[0])
			elev_y = int(elev_indices[1])
			print X, Y
			elev = elevationdata[elev_x][elev_y]
			m = slope[elev_x][elev_y]
			grade = aspect[elev_x][elev_y]
			chars.append(ex)
			chars.append(why)
			chars.append(NDVI)
			chars.append(tir1)
			chars.append(tir2)
			chars.append(elev)
			chars.append(m)
			chars.append(grade)
			chars.append(isBurn)
			writer.writerow(chars)
print float(burn/total)

im = toimage(zeroes)
im.save("test.jpg")
print thiscount




































