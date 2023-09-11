import torch.nn as nn
import torch


def train_seq_model(model,train_input,train_output, val_input, val_output, batch_train = 100, batch_val = 50,lr = 1e-1,epochs=50):

    criterion = nn.MSELoss()
    train_loss, val_loss, error = [],[],[]
    # training
    # epochs=list(range(epochs))
    optim = torch.optim.SGD(model.parameters(),lr = lr)

    for e in range(epochs):
        train_loss_=0
        error = torch.zeros(train_output.size(1))
        for j in range(0,train_input.size(0),batch_train):
        
            pred=model(train_input[j].view(-1))
        
            loss=criterion(pred,train_output[j])
            train_loss_+=loss.item()
            
            optim.zero_grad()
            loss.backward()
            optim.step()

        # print(train_loss_)
        # print(train_input.size(0))

        train_loss.append(train_loss_*batch_train/train_input.size(0)) #

        # print(train_loss)

        # validation 
        val_loss_=0

        for k in range(0,val_input.size(0),batch_val):
            valid=model(val_input[k].view(-1))
            val_loss_+=criterion(valid,val_output[k]).item()
            error += (valid-val_output[k]).abs()

        error_100 = ((error*batch_val/val_input.size(0))*100).round()
        print('Percentage error per feature on the validation set : ',error_100)
        
        val_loss.append(val_loss_*batch_val/val_input.size(0)) # 
        
    return train_loss, val_loss


def seq_model(features, neighbors = 1, nb_hidden=700,drop_rate = .5):
    '''this model contains 1 hidden layer'''
    assert isinstance(neighbors, int) and neighbors >= 1, 'provide positive integer value for neighbors. 1,2 recommended'
    assert isinstance(features, int) and features >= 1, 'provide positive integer value for features'
    assert isinstance(nb_hidden, int) and nb_hidden >= 1, 'provide positive integer value for nb_hidden'
    assert drop_rate < 1 and drop_rate > 0, 'Provide a drop_rate between 0 and 1'
    
    n_input = ((2*neighbors+1)**2-1)*features
    model = nn.Sequential(
        nn.Linear(n_input,nb_hidden)
        ,nn.Dropout(.5)
        ,nn.ReLU()
        ,nn.Linear(nb_hidden,features)
        ,nn.ReLU())
    
    return model

def seq_model_2(features, neighbors = 1, nb_hidden_1=700,nb_hidden_2=700,drop_rate = .5):
    ''' this model contains 2 hidden layers '''
    assert isinstance(neighbors, int) and neighbors >= 1, 'provide positive integer value for neighbors. 1,2 recommended'
    assert isinstance(features, int) and features >= 1, 'provide positive integer value for features'
    assert isinstance(nb_hidden_1, int) and nb_hidden_1 >= 1, 'provide positive integer value for nb_hidden_1'
    assert isinstance(nb_hidden_2, int) and nb_hidden_2 >= 1, 'provide positive integer value for nb_hidden_2'
    assert drop_rate < 1 and drop_rate > 0, 'Provide a drop_rate between 0 and 1'
    
    n_input = ((2*neighbors+1)**2-1)*features
    model = nn.Sequential(
        nn.Linear(n_input,nb_hidden_1,bias=True)
        ,nn.Dropout(drop_rate)
        ,nn.ReLU()
        ,nn.Linear(nb_hidden_1,nb_hidden_2,bias=True)
        ,nn.Dropout(drop_rate)
        ,nn.ReLU()
        ,nn.Linear(nb_hidden_2,features, bias=True)
        ,nn.ReLU())
    
    return model