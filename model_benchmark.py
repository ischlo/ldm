
# not necessary to test for all the combinations of values, try to select meaningful ones, run about 10-15 tests.

import os
import seq_model as sm
import data_engineering as de
from data_engineering import data_sets_t
from data_engineering import data_tensor
import matplotlib.pyplot as plt
import pandas as pd
import time

# pd.DataFrame(pd.Series(0,name='run')).to_csv('model_runs/run_num.csv',index=False)
# setting the run number, for traceability reasons. 

run_num = pd.read_csv('model_runs/run_num.csv')

# run_num

run_num=run_num.add(1)

# run_num

run_num.to_csv('model_runs/run_num.csv',index=False)

run_num = run_num['run'][0]

# THE PARAMETERS THAT ARE BENCHMARKED 

# FROM dropout paper, things to consider on a model with dropout
# - network size, n (fixed) units with probability p of dropout or n/p units based on some good value on n previously obtained ?
# - learning rate and momentum : large learning rate and momentum, especially for early epochs, 10-100 times greater l_r for dropout nets, .95-.99 momentum recommended for dropout
# - but then max-norm regularization with a c parameter. c can typically be around 3-4
# - vary the dropout itself. values around .55-.7 are found to work well. 

# model parameters

hidden_units = [900, 1100, 1500] #, 1100

neighborhood = [1,2]

hidden_layers = [1,2]

# worth or not ?
learn_rate = 2e-2 # [1e-1,5e-2,1e-2]
epochs = 100
train_batch = 200
val_batch = 60

# area = 'third'

## data set up

var_names = list(data_sets_t.keys())

print('Variables used: ', var_names)

features=len(var_names)

print('Training based on ', features,' features.')


error_rate = pd.DataFrame(pd.Series(range(epochs)))

start = time.time()

# data parameters
# extract cities from the rasters 
# for loops ?
for neighb in neighborhood:            
    train_input, train_output, val_input, val_output, test_input, test_output, coords = de.generate_data(data_tensor,n_neighbors=neighb)
    for unit in hidden_units:  
            for layer in hidden_layers:
                print('Model run with {} layer(s) : {} hidden unit(s), {}-neighbourhood. '.format(layer,unit,neighb))
                run_name = 'seq_{}_{}_{}'.format(layer,unit,neighb)
                print(('Running : '+ run_name))
                os.makedirs('model_runs/{}/seq_{}_{}_{}'.format(run_num,layer,unit,neighb))      
                if layer==1:
                    print('Using model with 1 hidden layer')
                    lu_seq = sm.seq_model(features=features,neighbors=neighb,nb_hidden=unit)
                elif layer==2:
                    print('Using model with 2 hidden layers')
                    lu_seq = sm.seq_model_2(features=features,neighbors=neighb,nb_hidden_1=unit,nb_hidden_2=unit,drop_rate=.5)
                else: 
                    print('Incorrectly provided number of hidden layers, using 1')
                    lu_seq = sm.seq_model(features=features,neighbors=neighb,nb_hidden=unit)

                train_loss,val_loss = sm.train_seq_model(model=lu_seq
                                                        ,train_input=train_input
                                                        ,train_output=train_output
                                                        ,val_input=val_input
                                                        ,val_output=val_output
                                                        ,epochs=epochs
                                                        ,batch_train=train_batch
                                                        ,batch_val=val_batch
                                                        ,lr=learn_rate
                                                        )

                error_rate[run_name+'_val']=val_loss
                error_rate[run_name+'_train']=train_loss
                
                plt.figure()
                plt.plot(list(range(epochs)),val_loss,label='Validation error')
                plt.plot(list(range(epochs)),train_loss, label='Training error')
                plt.xlabel('epoch')
                plt.ylabel('MSE')
                plt.title('Model calibration: {} layer(s), {} hidden units, {}-neighbourhood'.format(layer,unit,neighb))
                plt.legend()
                plt.savefig('model_runs/{}/{}/{}.png'.format(run_num,run_name,run_name))

t = time.time() - start

print('Execution time: {} seconds.'.format(t))

error_rate.to_csv('model_runs/{}/error_rate.csv'.format(run_num))
