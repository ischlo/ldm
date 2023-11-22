import torch.nn as nn
import torch
# import rioxarray
import xarray
import numpy as np
import data_engineering as de
from data_engineering import data_tensor
# from data_engineering import scale_factors
from data_engineering import data_sets_t
import first_net as fn
import matplotlib.pyplot as plt
from scipy.stats import entropy
import random


####

def plot_raster(data, title, dim = 0 ):
    if dim == -1:
        data = xarray.DataArray(data)
    else:
      data = xarray.DataArray(data[:,:,dim])

    f, ax = plt.subplots(figsize=(11, 4))
    data.plot()
    ax.title.set_size(20)
    ax.set(title=title)
    ax.axis('off')
    plt.show()  
####

var_names = list(data_sets_t.keys())

n_neighbors = 2

# res = de.get_cell(data_tensor,10,10,n_neighbors=n_neighbors)

# print(res.size(2))

# for i in range(res.size(2)):
#     plot_raster(data=res
#                 ,title=var_names[i]
#                 ,dim=i)
    
def entropy1(labels, base=None):
  value,counts = np.unique(labels, return_counts=True)
  return entropy(counts, base=base)

def cell_diversity(data, neighborhood = 3):
    '''this function takes as input a tensor of dim 3, 
    and computes the diversity of the input data along the last dimension
    by taking the entropy of all the data in the neighbourhood of each cell. 
    It should help identify the most diverse locations in terms of the data we have
      and help focus on these areas for further analysis.'''
    x_i = range(neighborhood,data.shape[1]-neighborhood)
    y_j = range(neighborhood,data.shape[2]-neighborhood)

    res = []

    for i in x_i:
        for j in y_j:
            x = de.get_cell(data,i,j,n_neighbors=neighborhood).reshape(-1).numpy()
            res.append(entropy(x))
    res=torch.tensor(res).view(data.shape[1]-2*neighborhood,data.shape[2]-2*neighborhood,1)
    res = (res-res.min())/(res.max()-res.min())

    return res

# test = cell_diversity(data=data_tensor,neighborhood=3)

plot_raster(data_sets_t['Education']
            ,title='Diversity of land use values, n=3'
            ,dim=-1)


# areas of interest contains the boundaries of the rectangles considered as 
# high diversity, > .9
# each element of the dict contains a list of size 4 with [ X_min, X_max, Y_min,Y_max]
areas_of_interest = {
    'first':[296,400,400,530]
    ,'second':[150, 200, 400,450]
    ,'third':[385,438,270,330]
    ,'fourth': [250,315,70,170]
}

# test[test>.9].shape

# mask = test > .9
# ind = torch.nonzero(mask)
# ind
# test
