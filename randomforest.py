import numpy as np
import pandas
from keras.layers import Dense, Activation
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import csv
import scipy.misc as sm
import cv2

raw = cv2.imread('/home/jayanthmouli/Desktop/automata/blank.jpg', cv2.IMREAD_GRAYSCALE)
#raw[:]=0
dataframe = pandas.read_csv("neighbors.csv").dropna()#.astype(np.float32)
colnames = ["NDVI", "TIR1", "TIR2", "elevation", "slope", "aspect"]
dataframe.drop_duplicates(subset=colnames, inplace=True)
dataset = dataframe.values
X = dataset[:,2:8]
y = dataset[:,9] #isBurnt
print dataframe.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 0)
regressor = RandomForestClassifier(n_estimators=20, random_state=0)  
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))

finalpred = regressor.predict(X_test)
n1 = 0
n2 = 0
for x in range(0,len(finalpred)):
	if finalpred[x] == y_test[x]:
		n1 = n1 + 1
	else:
		n2 = n2 + 1
totalpred = regressor.predict(X)
zeroes = np.zeros((7861, 7731,3),dtype=np.uint8)
print totalpred.shape
count = 0
for index, row in dataframe.iterrows():
	ndvi = float(row['NDVI'])
	tir1 = float(row['TIR1'])
	tir2 = float(row['TIR2'])
	elev = float(row['elevation'])
	slope = float(row['slope'])
	aspect = float(row['aspect'])
	prediction = regressor.predict([[ndvi, tir1, tir2, elev, slope, aspect]])
	xcoord = int(row['x'])
	ycoord = int(row['y'])
	print str(prediction[0])
	if prediction[0] == 1.0:
		
		print xcoord
		raw[ycoord][xcoord] = np.uint8(255)
print raw.shape
for d in range(len(raw)):
	for e in range(len(raw[d])):
		if raw[d][e] == 255:
			count = count + 1
print count
print raw[2232][2232]
cv2.imwrite("blank.jpg", raw)
cv2.namedWindow('zeroes',cv2.WINDOW_NORMAL)
cv2.resizeWindow('zeroes', 600,600)
cv2.imshow("zeroes", raw)
cv2.waitKey(0)
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
		
s = 'accuracy = ' + str(n1) + '/' + str(n1+n2)
plt.bar(['Correct Classification','Incorrect Classification'], [n1, n2])
plt.title('Random Forest Classification Results')
plt.text(0.5, 100, s , fontsize=12)
plt.savefig('classificationreport.png')
plt.show()
plt.plot(y_test, color = 'red', label = 'Real data')
plt.plot(y_pred, color = 'blue', label = 'Predicted data')
plt.title('Prediction')
plt.legend()