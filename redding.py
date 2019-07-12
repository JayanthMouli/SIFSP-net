import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
from osgeo import gdal
import os
import rasterio as rio
import utm
import scipy.misc as sm
import csv
import cv2
import richdem as rd

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
raw = cv2.imread('/media/jayanthmouli/54EC-046F/burned_area.jpg')
burned_img = raw[1572:3458, 3826:4808]
lowerbound = np.array([ 76,  26, 198])
upperbound = np.array([156, 106, 278])
burned = cv2.inRange(raw, lowerbound, upperbound)
with open("burnedindices.csv", "r") as csvinput:
	with open("whitepixels.csv", "w") as csvoutput:
		reader = csv.reader(csvinput)
		writer = csv.writer(csvoutput)
		d_reader = csv.DictReader(csvinput)
		headers = d_reader.fieldnames
		headers.append("NDVI")
		headers.append("TIR1")
		headers.append("TIR2")
		headers.append("elevation")
		headers.append("slope")
		headers.append("aspect")
		headers.append("windDiff")
		headers.append("isBurnt")
		writer.writerow(headers)
		for row in reader:
			neighbors = []
			x = int(row[0])
			y = int(row[1])
			neighbors.append([x,y])
			neighbors.append([x, y+1])
			neighbors.append([x+1, y+1])
			neighbors.append([x+1, y])
			neighbors.append([x+1, y-1])
			neighbors.append([x, y-1])
			neighbors.append([x-1, y-1])
			neighbors.append([x-1, y])
			neighbors.append([x-1, y+1])
			for neighbor in neighbors:
				chars = []
				X = int(neighbor[0])
				Y = int(neighbor[1])
				NDVI = ndvi[X][Y]
				tir1 = band10data[X][Y]
				tir2 = band11data[X][Y]
				UTM = pixel2coord(X,Y)
				coords = utm.to_latlon(UTM[0], UTM[1], 10, 'S')
				elev_indices = coord2pixel(coords[1], coords[0])
				elev_x = int(elev_indices[0])
				elev_y = int(elev_indices[1])
				elev = elevationdata[elev_x][elev_y]
				m = slope[elev_x][elev_y]
				grade = aspect[elev_x][elev_y]
				burn = isBurnt(X, Y, burned)
				windDiff = windangle(X, Y, x, y)
				chars.append(X)
				chars.append(Y)
				chars.append(NDVI)
				chars.append(tir1)
				chars.append(tir2)
				chars.append(elev)
				chars.append(m)
				chars.append(grade)
				chars.append(windDiff)
				chars.append(burn)
				writer.writerow(chars)
			
			
			
			
			
			
			
			
			
			
# fig, ax = plt.subplots(figsize=(10,6))
# ndvi_ret = ax.imshow(ndvi, cmap='PiYG', vmin=-1, vmax=1)
# fig.colorbar(ndvi_ret, fraction=.05)
# print ndvi[3458][1572]
# plt.show()



# for i in range(0,7861):
	# for j in range(0,7731):
		# band4_TOAR[i][j] = (band4data[i][j] * gain[4]) + bias[4]
		# band3_TOAR[i][j] = (band3data[i][j] * gain[3]) + bias[3]
		# band2_TOAR[i][j] = (band2data[i][j] * gain[2]) + bias[2]
def norm(band):
    band_min, band_max = band.min(), band.max()
    return ((band - band_min)/(band_max - band_min))
	

# RGB = np.dstack((norm(band7_TOAR), norm(band5_TOAR), norm(band3_TOAR)))

# sm.toimage(RGB,cmin=np.percentile(RGB,2), cmax=np.percentile(RGB,98)).save('falsecolor_python.tif')


# rgb = np.dstack((norm(band4_TOAR), norm(band3_TOAR), norm(band2_TOAR)))
# sm.toimage(rgb,cmin=np.percentile(rgb,2), cmax=np.percentile(rgb,98)).save('falsecolor_python.tif')
# fig, ax = plt.subplots(figsize=(10,6))
# ax.imshow(rgb)
# plt.show()

		
		
		
		
		
		
		
		
		
		

#fig, ax = plt.subplots(figsize=(10,6))
#ax.imshow(band3data)
#plt.show()
# print band3.RasterXSize
# print band3.RasterYSize


 
rows = 7731
colms = 7861

# if __name__ == "__main__":
 # test1 =  pixel2coord(456, 345)
 # test = utm.to_latlon(test1[0], test1[1], 10, 'S')
 # print test
 # coords = coord2pixel(test[1], test[0])
 # print elevationdata[int(coords[0])][int(coords[1])]
 

 
   
   
   
   
   


# for line in fp: # 
      # # Check for LMAX and LMIN strings
      # # Note that parse logic is identical to the first case
      # # This version of the code works, but is rather inelegant!
    # if ( line.find ("RADIANCE_MULT_BAND") >= 0 ):
        # s = line.split("=") # Split by equal sign
        # the_band = int(s[0].split("_")[3]) # Band number as integer
        # gain[the_band] = float ( s[-1] ) # Get constant as float
    # elif ( line.find ("RADIANCE_ADD_BAND") >= 0 ):
        # s = line.split("=") # Split by equal sign
        # the_band = int(s[0].split("_")[3]) # Band number as integer
        # bias[the_band] = float ( s[-1] ) # Get constant as float
    # elif ( line.find ("QUANTIZE_CAL_MAX_BAND") >= 0 ):
        # s = line.split("=") # Split by equal sign
        # the_band = int(s[0].split("_")[4]) # Band number as integer
        # qc_lmax[the_band] = float ( s[-1] ) # Get constant as float
    # elif ( line.find ("QUANTIZE_CAL_MIN_BAND") >= 0 ):
        # s = line.split("=") # Split by equal sign
        # the_band = int(s[0].split("_")[4]) # Band number as integer
        # qc_lmin[the_band] = float ( s[-1] ) # Get constant as float
    # elif ( line.find ("RADIANCE_MAXIMUM_BAND") >= 0 ):
        # s = line.split("=") # Split by equal sign
        # the_band = int(s[0].split("_")[3]) # Band number as integer
        # lmax[the_band] = float ( s[-1] ) # Get constant as float
    # elif ( line.find ("RADIANCE_MINIMUM_BAND") >= 0 ):
        # s = line.split("=") # Split by equal sign
        # the_band = int(s[0].split("_")[3]) # Band number as integer
        # lmin[the_band] = float ( s[-1] ) # Get constant as float
# for x in gain:
	# print x
	



