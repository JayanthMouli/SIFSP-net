import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
from osgeo import gdal
import os
import rasterio as rio
import utm
import scipy.misc as sm
import cv2

count = 0

raw = cv2.imread('/media/jayanthmouli/54EC-046F/burned_area.jpg')
zeroes = np.zeros((7861, 7731))
burned_img = raw[1572:3458, 3826:4808]
lowerbound = np.array([ 76,  26, 198])
upperbound = np.array([156, 106, 278])
burned = cv2.inRange(raw, lowerbound, upperbound)
cv2.imwrite("burned.jpg", burned)
burned_indices = np.where(burned!=[0])
print max(burned_indices[1])
with open("burnedindices.csv", "w") as csvoutput:
	writer = csv.writer(csvoutput)
	header = ["x", "y"]
	writer.writerow(header)
	for x in range(len(burned_indices[0])):
		if((3825<int(burned_indices[0][x])<4808) and (1572<int(burned_indices[1][x])<3458)):
			ary = []
			ary.append(burned_indices[0][x])
			ary.append(burned_indices[1][x])
			x_value = burned_indices[0][x]
			y_value = burned_indices[1][x]
			zeroes[y_value][x_value] = 255			
			writer.writerow(ary)
			
cv2.namedWindow('zeroes',cv2.WINDOW_NORMAL)
cv2.resizeWindow('zeroes', 600,600)
cv2.imshow("zeroes", zeroes)
cv2.waitKey(0)




for x in range(len(zeroes)):
	for y in range(len(zeroes[x])):
		if zeroes[x][y] == 255:
			count +=1
print count