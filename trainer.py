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

dataframe = pandas.read_csv("trainer.csv").dropna()#.astype(np.float32)
# colnames = ["NDVI", "TIR1", "TIR2", "elevation", "slope", "aspect"]
# dataframe.drop_duplicates(subset=colnames, inplace=True)
dataset = dataframe.values
X = dataset[:,2:8]
y = dataset[:,8] #isBurnt


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 0)
regressor = RandomForestClassifier(n_estimators=20, random_state=0)  
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))

zeroes = Image.open('burnedcut.jpg')
pix = zeroes.load()
width, height = zeroes.size
new = np.zeros((height, width))

with open("blackindicesmod.csv", "r") as csvinput:
	reader = csv.reader(csvinput)
	for row in reader:
		x = int(row[0])-3826
		y = int(row[1]) - 1572
		ndvi = float(row[2])
		tir1 = float(row[3])
		tir2 = float(row[4])
		elev = float(row[5])
		slope = float(row[6])
		grade = float(row[7])
		seed = regressor.predict([[ndvi, tir1, tir2, elev, slope, grade]])
		if int(seed) == 1:
			new[y][x] = 255
		else:
			new[y][x] = 0
cv2.imwrite('prediction.jpg', new)
		
			