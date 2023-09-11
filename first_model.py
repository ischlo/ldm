import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import data_engineering as de
from data_engineering import data_tensor
from data_engineering import data_sets_t
import first_net as fn
import matplotlib.pyplot as plt
import seq_model as sm
from de_debugging import areas_of_interest
import random as rn
import xarray
from de_debugging import plot_raster
from de_debugging import cell_diversity

var_names = list(data_sets_t.keys())

print('Variables used: ', var_names)

features=len(var_names)

print('Training based on ', features,' features.')

id = list(range(0,len(var_names)))

var_id=dict(zip(var_names,id))

#### initiating the data and model

def sub_area(data, area):
    bounds = areas_of_interest[area]
    return data[:,bounds[2]:bounds[3],bounds[0]:bounds[1],:]

area_data = sub_area(data_tensor, 'third')
# print('The shape of the training output is : ',train_output.shape)
# print('The shape of the validation output is : ',val_output.shape)

# plot_raster(cell_diversity(area_data,neighborhood=1),title='Diversity of area',dim=-1)

# parameters

nb_hidden= 900
epochs = 150
batch_train = 1
batch_val = 1
nb_layers = 1
neighbors = 1

# Formating the train, validation, test data. 

train_input, train_output, val_input, val_output, test_input, test_output, coords = de.generate_data(area_data,n_neighbors=neighbors,size=.9)

# train_input.shape

if nb_layers==1 :
    print('Using model with 1 hidden layer')
    lu_seq = sm.seq_model(features=features,neighbors=neighbors,nb_hidden=nb_hidden)
elif nb_layers==2 :
    print('Using model with 2 hidden layers')
    lu_seq = sm.seq_model_2(features=features,neighbors=neighbors,nb_hidden_1=nb_hidden,nb_hidden_2=nb_hidden,drop_rate=.5)
else : 
    print('Incorrectly provided number of hidden layers, using 1')
    lu_seq = sm.seq_model(features=features,neighbors=neighbors,nb_hidden=nb_hidden)

train_loss,val_loss = sm.train_seq_model(model=lu_seq
                                         ,train_input=train_input
                                         ,train_output=train_output
                                         ,val_input=val_input
                                         ,val_output=val_output
                                         ,epochs=epochs
                                         ,batch_train=batch_train
                                         ,batch_val=batch_val
                                         ,lr=2e-2)

# rn = [100,200,300,1000,1500]

# for i in rn:
#     print(pd.DataFrame(list(zip(var_names,lu_seq(val_input[i].view(-1)).tolist(),val_output[i].tolist())),columns=['Variable','model_output','model_input']))

plt.plot(list(range(epochs)),val_loss,label='Validation error')
plt.plot(list(range(epochs)),train_loss, label='Training error')
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.title('Model calibration: {}-neighbourhood, {}-hidden units, {}-layers.'.format(neighbors,nb_hidden,nb_layers))
plt.legend()
plt.show()
# plt.savefig('seq_{}_{}_{}.png'.format(nb_layers,nb_hidden,neighbors))

# for i in range()
#  Poissibility to start from here by loading the last trained and saved model
# torch.save(lu_seq.state_dict(), 'lu_seq_model_{}_{}_{}.pt'.format(nb_hidden,nb_layers,neighbors))
# 
# loading a models parameters into a model defined in the script with the matching architecture. 
# lu_seq = sm.seq_model(features=features,neighbors=neighbors,nb_hidden=nb_hidden)
# lu_seq.load_state_dict(torch.load('lu_seq_model_900_1_1.pt'))
# ## check what was loaded
# lu_seq.eval()


area_test = sub_area(data_tensor, 'first')

area_test_in,area_test_out,coords_test = de.generate_data(area_test,n_neighbors=neighbors,size=1)

simulation_error = np.zeros((area_test.shape[1],area_test.shape[2]))

for i,el in enumerate(coords_test):
    sim = lu_seq(area_test_in[i].view(-1))
    error = torch.sum(area_test_out[i]-sim).item()
    simulation_error[el[0],el[1]]= error
# for i,el in enumerate(coords_test['val_coords']):
#     sim = lu_seq(area_test_in[i].view(-1))
#     error = torch.sum(area_test_out[i]-sim).item()
#     simulation_error[el[0],el[1]]= error
# for i,el in enumerate(coords_test['train_coords']):
#     sim = lu_seq(area_test_in[i].view(-1))
#     error = torch.sum(area_test_out[i]-sim).item()
#     simulation_error[el[0],el[1]]= error

simulation_error.max()

def plot_array(data, title):
    data = xarray.DataArray(data)
    f, ax = plt.subplots(figsize=(11, 4))
    data.plot()
    ax.title.set_size(20)
    ax.set(title=title)
    ax.axis('off')
    plt.show() 

plot_array(simulation_error,title='Error on region 1 from training on region 3')

plot_raster(cell_diversity(area_test,neighborhood=1),title='Diversity of area',dim=-1)