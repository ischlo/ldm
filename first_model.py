import torch.nn as nn
import torch
import numpy as np
import data_engineering as de
from data_engineering import data_tensor
from data_engineering import scale_factors
import first_net as fn
import matplotlib.pyplot as plt
# import torch
import seq_model as sm

# train_input, train_output, validate_input, validate_output = de.lu_data(n_neighbors = 1)
var_names = [ 
    # variable taking values of 0 and 1 only. 
    'greenbelt'
    ,'flood100'
    ,'bound'
    ,'surface_water'
    ,'heritage'
    ,'conservation'
    ,'PnG'
    ,'AONB'
    ,'highway'
    ,'SSSI'
    ,'nature_reserves'
    ### 'Continuous' variables next
    ,'slope'
    ,'population'
    # ,'gs_2500'
    ,'job1'
    # ,'job2'
    # ,'job3'
    # ,'job4'
    # ,'job5'
    # ,'job6'
    ,'house1'
    # ,'house2'
    # ,'house3'
    # ,'house4'
    # ,'house5'
    # ,'house6'
]

features=len(var_names)

print('Training based on ', features,' features.')

id = list(range(0,len(var_names)))

var_id=dict(zip(var_names,id))

###3 initiating the data and model

neighbors = 1

train_input, train_output, val_input, val_output, test_input, test_output = de.generate_data(data_tensor,n_neighbors=neighbors)

print('The shape of the training output is : ',train_output.shape)
print('The shape of the validation output is : ',val_output.shape)

# model parameters

nb_hidden= 900
nb_epochs = 100

epochs=list(range(nb_epochs))

print('The shape of the training data is : ',train_input.shape)

lu_seq = sm.seq_model(features=features,neighbors=neighbors,nb_hidden=nb_hidden)

train_loss , val_loss , epochs = sm.train_seq_model(model=lu_seq
                                                    ,train_input=train_input
                                                    ,train_output=train_output
                                                    ,val_input=val_input
                                                    ,val_output=val_output
                                                    ,epochs=nb_epochs
                                                    ,lr=5e-2
                                                    )

plt.plot(epochs,val_loss,label='Validation error')
plt.plot(epochs,train_loss, label='Training error')
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.title('Model calibration with {}-neighbourhood and {} hidden units.'.format(neighbors,nb_hidden))
plt.legend()
plt.show()