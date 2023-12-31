#  this script will take the bunch of rasters
# transform them into one big tensor, 
#  and define functions to sample this tensor into train, validate and test sets.
import random
import torch
import math

# READ IN THE DATA HERE. 

#  for example by calling the reading rtaters script. 
from py.reading_rasters import data_sets

# print(data_sets)

# copying it into another dictionary that will hold tensors.
data_sets_t = data_sets.copy()

scale_factors = []

cont_vars = ['slope'
             ,'population'
             ,'job1'
             ,'job2'
             ,'job3'
             ,'job4'
             ,'job5'
             ,'job6'
             ,'house1'
             ,'house2'
             ,'house3'
             ,'house4'
             ,'house5'
             ,'house6'
             ]


# transforming numpy array into tensors 
for x in data_sets_t:
   # normalizing by max
   # print(x)
   data_sets_t[x] = torch.from_numpy(data_sets_t[x].to_numpy()/data_sets_t[x].to_numpy().max())
   # # normalizing by mean and std
   # if x in cont_vars:
   #    data_sets_t[x] = torch.from_numpy((data_sets_t[x].to_numpy()-data_sets_t[x].to_numpy().mean())/data_sets_t[x].to_numpy().std())
   # else : 
   #     data_sets_t[x] = torch.from_numpy(data_sets_t[x].to_numpy())
   # scale_factors.append(data_sets_t[x].to_numpy().max())

#  what is the right tensor format ?
data_tensor = torch.stack(tuple(data_sets_t.values()),-1)

#  function that takes as parameter the size of the neighbourhood of a cell to take, 
# and a size for the train sample, and returns the corresponding train, valid, test data sets. 

def sample_cell(data, n_neighbors=1):
   
   rand_x = random.randint(n_neighbors,data.size(1)-n_neighbors-1)
   rand_y = random.randint(n_neighbors,data.size(2)-n_neighbors-1)
   return data[data.size(0)-1,rand_x-n_neighbors:rand_x+n_neighbors+1,rand_y-n_neighbors:rand_y+n_neighbors+1]

def get_cell(data,x,y,n_neighbors=1):
   '''gets a cell a the specified x and y coordinates as well as the neighbourhood.'''
   return data[0,x-n_neighbors:x+n_neighbors+1,y-n_neighbors:y+n_neighbors+1]
   
def get_input(data,n_neighbors=1,constrained = False):
   '''generates input data from a 3d tensor for the case when we are considering neighbourhoods.
   Removes the central element basically that is predicted. '''
   x_i = range(0,data.shape[1])
   y_j = range(0,data.shape[2])
   res = []
   for i in x_i:
      for j in y_j:
         if constrained:
            res.append(data[:,i,j])
         elif i!=n_neighbors or j!=n_neighbors:
            res.append(data[:,i,j])
   return torch.stack(res,1).float()


# could be a simple function that returns the central cell of a neighbourhood as target.  
# def get_target(data,x,y):

# for i in range(0,data_tensor.size(-1)):
#    print(data_tensor[0,:,:,i].max())

# test_samp=sample_cell(data_tensor,2)
# test_samp.size()

# this function takes the data_tensor and divides it into disjoint train, validation and testing data sets
# size is the fraction of the data to be used for the train set, and half of the rest is taken per other data set
def generate_data(data, size = .7,n_neighbors = 1, constraint_keys = None):
   '''Generate training, testing, validating data sets.'''

   constraints_input = []
   if constraint_keys is not None:
      keys = [k.lower() for k in data_sets_t.keys()]
      constraint_keys=[x.lower() for x in constraint_keys]
      assert all([x.lower() in keys for x in constraint_keys]), 'provide constraints that exist'
      constraints_input = set([keys.index(x) for x in constraint_keys])

   var_output = set(range(len(data_sets_t)))
   var_output.difference_update(constraints_input)

   #turning back into a list containing the indices of the layers of interest
   var_output=list(var_output)
   constraints_input=list(constraints_input)

   #### 
   print('Constraints of the model at index : ', constraints_input)
   print('Predicted variables at ',var_output)

   whole,train,test,val,coords = [],[],[],[],[]

   rand_1 = list(range(n_neighbors,data.size(1)-n_neighbors))
   rand_2 = list(range(n_neighbors,data.size(2)-n_neighbors))
   
   random.shuffle(rand_1)
   random.shuffle(rand_2)

   coords = []# list(zip(rand_1,rand_2))

   # print('entering the train loop')
   for i in rand_1:
      for j in rand_2:
         whole.append(get_cell(data,i,j,n_neighbors))
         coords.append((i,j))

   if size==1: 
      whole=torch.stack(whole,0)
      whole_output = whole[:,n_neighbors,n_neighbors].float()
      whole_input=get_input(whole,n_neighbors=n_neighbors)

      return whole_input,whole_output, coords
   elif size<1:
      size = int(len(whole) * size)

      if (len(whole)-size)%2 != 0:
         size-=1

      size_2 = int((len(whole)+size)/2)

      train = whole[:size]
      val = whole[size:size_2]
      test = whole[size_2:len(whole)]

      coords_dict = {'train_coords' : coords[:size]
                     ,'val_coords': coords[size:size_2]
                     ,'test_coords': coords[size_2:len(coords)]
                     }

      train= torch.stack(train,0)
      val= torch.stack(val,0)
      test= torch.stack(test,0)

      print('Train set : {}'.format(train.shape))
      print('Validation set: {}'.format(val.shape))
      print('Test set: {}'.format(test.shape))

      #  generating the output data
      # if constraints are provided, keep them as input

      if constraint_keys is not None:

         train_output = train[:,n_neighbors,n_neighbors,var_output].float()
         val_output = val[:,n_neighbors,n_neighbors,var_output].float()
         test_output = test[:,n_neighbors,n_neighbors,var_output].float()   

         # generating the inputs 
         train_input = get_input(train[:,:,:,constraints_input], n_neighbors=n_neighbors,constrained = True)
         val_input = get_input(val[:,:,:,constraints_input], n_neighbors=n_neighbors,constrained = True)
         test_input = get_input(test[:,:,:,constraints_input], n_neighbors=n_neighbors,constrained = True)

         return train_input, train_output, val_input,val_output,test_input,test_output, coords_dict

      else:
         train_output = train[:,n_neighbors,n_neighbors].float()
         val_output = val[:,n_neighbors,n_neighbors].float()
         test_output = test[:,n_neighbors,n_neighbors].float()

         # generating the inputs 
         train_input = get_input(train, n_neighbors=n_neighbors)
         val_input = get_input(val, n_neighbors=n_neighbors)
         test_input = get_input(test, n_neighbors=n_neighbors)

         return train_input, train_output, val_input,val_output,test_input,test_output, coords_dict
