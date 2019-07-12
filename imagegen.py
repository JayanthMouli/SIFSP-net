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

ary = np.zeros((7861, 7731))

raw = cv2.imread('burned.jpg',0)
newRaw = raw
dataframe = pandas.read_csv("neighbors.csv").dropna()#.astype(np.float32)
colnames = ["NDVI", "TIR1", "TIR2", "elevation", "slope", "aspect"]
dataframe.drop_duplicates(subset=colnames, inplace=True)
dataset = dataframe.values
X = dataset[:,2:8]
y = dataset[:,9] #isBurnt


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 0)
regressor = RandomForestClassifier(n_estimators=20, random_state=0)  
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))

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
total = 0
zeroes = Image.open('macro.jpg')
pix = zeroes.load()
width, height = zeroes.size
new = np.zeros((height, width))
cv2.namedWindow('inprogress',cv2.WINDOW_NORMAL)
cv2.resizeWindow('inprogress', 600,600)




with open('blackindices.csv', 'r') as csvinput:
	with open('blackindicesmod.csv', 'w') as csvoutput:
		reader = csv.reader(csvinput)
		reader.next()
		writer = csv.writer(csvoutput)
		for row in reader:
			chars = []
			count = 0
			ex = int(row[1])
			why = int(row[0])
			x = ex+3826
			y = why + 1572
			localNDVI = ndvi[x][y]
			tir1 = band10data[x][y]
			tir2 = band11data[x][y]
			UTM = pixel2coord(x,y)
			coords = utm.to_latlon(UTM[0], UTM[1], 10, 'S')
			elev_indices = coord2pixel(coords[1], coords[0])
			elev_x = int(elev_indices[0])
			elev_y = int(elev_indices[1])
			elev = elevationdata[elev_x][elev_y]
			m = slope[elev_x][elev_y]
			grade = aspect[elev_x][elev_y]
			seed = regressor.predict([[localNDVI, tir1, tir2, elev, m, grade]])
			chars.append(x)
			chars.append(y)
			chars.append(localNDVI)
			chars.append(tir1)
			chars.append(tir2)
			chars.append(elev)
			chars.append(m)
			chars.append(grade)
			writer.writerow(chars)
			print x, y, seed
			if int(seed) == 1:
				print count
				new[why][ex] = 255
			else:
				new[why][ex] = 0
			print total
cv2.imwrite('burnedprediction.jpg', new)

		
cv2.namedWindow('burn',cv2.WINDOW_NORMAL)
cv2.resizeWindow('burn', 600,600)
cv2.imshow("burn", new)
cv2.waitKey(0)


	