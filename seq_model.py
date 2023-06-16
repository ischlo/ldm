import torch.nn as nn
import torch


def train_seq_model(model,train_input,train_output, val_input, val_output, batch_train = 100, batch_val = 50,lr = 1e-1,epochs=50):

    criterion = nn.MSELoss()
    train_loss, val_loss, error = [],[],[]
    # training
    # epochs=list(range(epochs))

    for e in range(epochs):

        train_loss_=0
        error = torch.zeros(train_output.size(1))

        for j in range(0,train_input.size(0),batch_train):
                    
            pred=model(train_input[j].view(-1))
            
            
        
            loss=criterion(pred,train_output[j])
            train_loss_+=loss.item()
            
            optim = torch.optim.SGD(model.parameters(),lr = 1e-1)
            optim.zero_grad()
            loss.backward()
            optim.step()
        train_loss.append(train_loss_/train_input.size(0)/batch_train)

        # validation 
        val_loss_=0

        for k in range(0,val_input.size(0),batch_val):
            valid=model(val_input[k].view(-1))
            val_loss_+=criterion(valid,val_output[k]).item()
            error += (valid-train_output[j]).abs()

        error_100 = (error*batch_val/val_input.size(0)*100).round()
        print('Percentage error per feature on the validation set : ',error_100)
        
        val_loss.append(val_loss_/val_input.size(0)/batch_val)
        
    return train_loss, val_loss, list(range(epochs))


def seq_model(features, neighbors = 1, nb_hidden=700,drop_rate = .5):

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