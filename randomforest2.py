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
burned = cv2.inRange(burned_img, lowerbound, upperbound)		

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


zeroes = np.zeros((7861, 7731))



# with open('burnedindices.csv', 'r') as csvinput:
	# reader = csv.reader(csvinput)
	# reader.next()
	# for row in reader:
		# x_index = int(row[0])
		# y_index = int(row[1])
		# localNDVI = ndvi[x_index][y_index]
		# tir1 = band10data[x_index][y_index]
		# tir2 = band11data[x_index][y_index]
		# UTM = pixel2coord(x_index,y_index)
		# coords = utm.to_latlon(UTM[0], UTM[1], 10, 'S')
		# elev_indices = coord2pixel(coords[1], coords[0])
		# elev_x = int(elev_indices[0])
		# elev_y = int(elev_indices[1])
		# elev = elevationdata[elev_x][elev_y]
		# m = slope[elev_x][elev_y]
		# grade = aspect[elev_x][elev_y]
		# seed = regressor.predict([[localNDVI, tir1, tir2, elev, m, grade]])
		# if int(seed) == 1:
			# zeroes[x_index][y_index] = 255
		# else:
			# zeroes[x_index][y_index] = 0		
	
total = 0
subtotal = 0
threshold = cv2.imread('burned.jpg')
for x_ind in range(len(burned)):
	for y_ind in range(len(burned[x_ind])):
		if burned[x_ind][y_ind] ==255:
			total += 1
			x = int(x_ind) + 3825
			y = int(y_ind) + 1572
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
			if int(seed) == 1:
				zeroes[x][y] = 255
				subtotal +=1
			else:
				zeroes[x][y] = 0
			print total
print type(burned[24][24]), threshold[234][234], threshold[543][543]
	
	

new_p = Image.fromarray(zeroes)
if new_p.mode != 'RGB':
    new_p = new_p.convert('RGB')
new_p.save('blank.jpg')

# for index, row in dataframe.iterrows():
	# ndvi = float(row['NDVI'])
	# tir1 = float(row['TIR1'])
	# tir2 = float(row['TIR2'])
	# elev = float(row['elevation'])
	# slope = float(row['slope'])
	# aspect = float(row['aspect'])
	# prediction = regressor.predict([[ndvi, tir1, tir2, elev, slope, aspect]])
	# xcoord = int(row['x'])
	# ycoord = int(row['y'])
	# print str(prediction[0])
	# if prediction[0] == 1.0:
		# zeroes[xcoord][ycoord] = 255

# new_p = Image.fromarray(zeroes)
# if new_p.mode != 'RGB':
    # new_p = new_p.convert('RGB')
# new_p.save('blank.jpg')
# predimage = cv2.imread('blank.jpg')
# print dataframe.shape
# ROI = predimage[1572:3458, 3826:4808]
# cv2.imshow("ROI", ROI)
# cv2.waitKey(0)

# fig, ax = plt.subplots(figsize=(10,6))
# ax.imshow(zeroes)
# plt.show()
# with open("randomforesttest.csv", "w") as csvoutput:
	# writer = csv.writer(csvoutput)
	# headers = []
	# headers.append("pred")
	# headers.append("actual")
	# writer.writerow(headers)
	# for x in range(0,500):
		# ary = []
		# ary.append(finalpred[x])
		# ary.append(y_test[x])
		# writer.writerow(ary)
		
# s = 'accuracy = ' + str(n1) + '/' + str(n1+n2)
# plt.bar(['Correct Classification','Incorrect Classification'], [n1, n2])
# plt.title('Random Forest Classification Results')
# plt.text(0.5, 100, s , fontsize=12)
# plt.savefig('classificationreport.png')
# plt.show()
# plt.plot(y_test, color = 'red', label = 'Real data')
# plt.plot(y_pred, color = 'blue', label = 'Predicted data')
# plt.title('Prediction')
# plt.legend()