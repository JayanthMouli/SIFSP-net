# SIFSP-net
Satellite Imagery Fire Shape Prediction Network (SISFPNET) is a machine learning approach to 
predicting wildfire shape based on satellite imagery. Data are from NASA's Landsat 8 mission, 
which collected satellite imagery of the Earth throughout 2018. Specifically, SIFSPNET uses
Landsat OLI/TIRS data for geospatial and thermal imagery, and treats each image as a 2 dimensional matrix of pixels. The network assigns characteristics
to each pixel in the satellite image; these variables include temperature (derived from TIR sensors), 
elevation (derived from Aster DEM data), slope, aspect, NDVI (Normalized Difference Vegetation Index) and wind difference*. 
The burned area of the wildfire is determined by constructing a false-color image by combining Shortwave Infrared, Near Infrared,
and green Landsat bands. The image is then thresholded to determine which pixels are burnt. SISFPNET is a random forest classifier neural network. SISFPNET is trained with a portion of
the pixels in the satellite image with the said pixel characteristics. Once the coordinate of the origin of the wildfire is inputted, 
SISFPNET simulates wildfire spread by creating a new 2-D matrix, where pixels are classified as either not burnt, burnt, or currently burning. Once SISFPNET
finishes the simulation, the predicted burned area is saved as a JPEG image. 

SISFPNET was tested primarily with data from the Carr Fire in July/August 2018 near Redding, California. This fire had notably strong southeasterly 
winds that primarily contributed to fire spread. Future work with SISFPNET aims to improve wind analysis techniques, as well as incorportate temporal data by
modeling wildfire spread as a function of time.


* Wind difference indicates the difference between a pixel's theoretical fire spread direction and the prominent wind direction in the area. 


