# FIRST MODEL OF DEEP LEARNING ARCHITECTURE FOR LAND USE 
from torch import nn
import torch
import torch.nn.functional as F

# function  that based on the desired neighbourhood size, and number of features to predict, 
# returns the reaquired size of the hidden layers linear operations. 

def n_input(n, features=26):
    return ((2*n+1)**2-1)*features



class first_net(nn.Module):
  def __init__(self,n,features=26,nb_hidden=500):
    super().__init__()
    self.l1=nn.Linear(((2*n+1)**2-1)*features,nb_hidden)
    self.l2=nn.Linear(nb_hidden,features)

  def forward(self,x):
     x=x.float()
     x=F.relu(self.l1(x.view(-1)))
     x=F.relu(self.l2(x))
     return x
  
def train_model(model,train_input,train_target,mini_batch_size=100,nb_epochs=1):
    criterion=nn.CrossEntropyLoss()
    eta=1e-1
    for e in range(nb_epochs):
        acc_loss = 0
        print('starting the batch treatment')
        # We do this with mini-batches
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            print('model done')
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            print('loss computed')
            print(loss)
            acc_loss = acc_loss + loss.item()

            model.zero_grad()
            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -= eta * p.grad

        print(e, acc_loss)




