import richdem as rd
import numpy as np
import matplotlib.pyplot as plt

dem = rd.LoadGDAL("/media/jayanthmouli/54EC-046F/elevation/the_better_one.tif", no_data = 0)
slope = rd.TerrainAttribute(dem, attrib='slope_riserun')
# rd.rdShow(slope, axes=False, cmap='magma', figsize=(8, 5.5))
# plt.show()

aspect = rd.TerrainAttribute(dem, attrib='aspect')
rd.rdShow(aspect, axes=False, cmap='jet', figsize=(8, 5.5))
plt.show()
print aspect.shape
