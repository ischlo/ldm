import torch 
import rioxarray
import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
sys.path.append('./py')
import random
from config_OXF import inputs, outputs

### THIS SCRIPT INITIALIZES THE TENSORS FOR THE LEARNING

# WE GET A SET OF RASTERS, NORMALIZE THEM, COMBINE THEM TOGETHER INTO ONE RASTER GRID.
# THEN RANDOMLY SAMPLE N_s ELEMENTS FROM WHITHIN THE GRID (AT LEAST N_n STEPS FROM THE BOUNDARY)
# AND ADD THE N_n NEIGHBORS TO IT. 


# adding the right file path to the original from github 
for x in inputs:
  inputs[x] = 'py/'+ inputs[x]

# print(inputs)
# initializing a dictionary with all the data sets
data_sets = {
    # 'greenbelt' : rioxarray.open_rasterio(inputs["Greenbelt"])
    # ,
    # 'flood100' : rioxarray.open_rasterio(inputs["Floodzone100"])
    'flood1000' : rioxarray.open_rasterio(inputs["Floodzone1000"])
    # ,'bound' : rioxarray.open_rasterio(inputs["OxfordBoundary"])
    # ,'surface_water' : rioxarray.open_rasterio(inputs["SurfaceWater"])
    ,'heritage' : rioxarray.open_rasterio(inputs["HistoricalConservation"])
    ,'conservation' : rioxarray.open_rasterio(inputs["ConservationArea"])
    ,'PnG' : rioxarray.open_rasterio(inputs["ParksAndGardens"])
    ,'AONB' : rioxarray.open_rasterio(inputs["AONB"])
    # ,'highway' : rioxarray.open_rasterio(inputs["Highways"])
    # ,'SSSI' : rioxarray.open_rasterio(inputs["SSSI"])
    # ,'nature_reserves' : rioxarray.open_rasterio(inputs["NNR"])
    #####
    ,'slope' : rioxarray.open_rasterio(inputs["Slope"])
    ,'population' : rioxarray.open_rasterio(inputs["Population"])
    # ,'gs_2500' : rioxarray.open_rasterio(inputs["GS2500"])
    ,'job1' : rioxarray.open_rasterio(inputs["JobAccessibilityRoads"])
    ,'job2' : rioxarray.open_rasterio(inputs["JobAccessibilityBus"])
    ,'job3' : rioxarray.open_rasterio(inputs["JobAccessibilityRail"])
    ,'job4' : rioxarray.open_rasterio(inputs["JobAccessibilityRoads2030"])
    ,'job5' : rioxarray.open_rasterio(inputs["JobAccessibilityBus2030"])
    ,'job6' : rioxarray.open_rasterio(inputs["JobAccessibilityRail2030"])
    ,'house1' : rioxarray.open_rasterio(inputs["HousingAccessibilityRoads"])
    ,'house2' : rioxarray.open_rasterio(inputs["HousingAccessibilityBus"])
    ,'house3' : rioxarray.open_rasterio(inputs["HousingAccessibilityRail"])
    ,'house4' : rioxarray.open_rasterio(inputs["HousingAccessibilityRoads2030"])
    ,'house5' : rioxarray.open_rasterio(inputs["HousingAccessibilityBus2030"])
    ,'house6' : rioxarray.open_rasterio(inputs["HousingAccessibilityRail2030"])
} 

# print(data_sets['flood1000'])

# check the data visualisation
# f = plt.figure()
# f.set_size_inches(8, 6)
# ax2 = plt.subplot(1, 1, 1)  # columns, rows, location
# data_sets['greenbelt'].plot(ax=ax2, cmap="Blues", vmin=0, vmax=1)
# 
# scalebar = ScaleBar(1, "m", length_fraction=0.25, location='lower right',scale_loc='top',box_color=None)
# ax2.add_artist(scalebar)
# x, y, arrow_length = 0.05, 0.95, 0.06
# ax2.annotate('N', xy=(x, y), xytext=(x, y - arrow_length), arrowprops=dict(facecolor='black', width=2, headwidth=8), ha='center', va='center', fontsize=15, xycoords=ax2.transAxes)
# plt.title("GB_FBA_LD_2030", fontsize=14, color='black', loc='center', pad=16)
# plt.xticks([])
# plt.yticks([])
# plt.xlabel("")
# ax2.yaxis.set_label_position("right")
# plt.ylabel("Development Potential", fontsize=14, labelpad=10)
# f.tight_layout()
# 



