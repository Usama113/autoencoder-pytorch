import numpy as np
import pandas
import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
from torch.utils.data.sampler import RandomSampler,SequentialSampler
from torch.optim.lr_scheduler import MultiStepLR



class autoencoder(nn.Module):
    def __init__(self,input_dim):
        super(autoencoder, self).__init__()
        self.input_dim=input_dim
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 250),
            nn.ReLU(),
            nn.Linear(250, 200),
            nn.ReLU(), nn.Linear(200, 50))
        self.decoder = nn.Sequential(
            nn.Linear(50, 200),
            nn.ReLU(),
            nn.Linear(200, 250),
            nn.ReLU(), nn.Linear(250, self.input_dim))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self,X):
        en=self.encoder(X)
        return en



class custom_dataset(Dataset):
    def __init__(self,data):
        self.data=data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.Tensor(self.data[idx])
def autoencoder_setup(X):

    criterion = nn.MSELoss()
    # criterion=nn.L1Loss()
    input_dim=X.shape[1]
    net = autoencoder(input_dim)

    ds = custom_dataset(X)
    Rsampler=RandomSampler(ds)
    dl = DataLoader(ds, batch_size=200,sampler=Rsampler)

    learning_rate = 0.001
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    #scheduler = MultiStepLR(optimizer, milestones=[70, 120], gamma=0.01)

    i = 0
    num_epochs = 1000
    loss_min=-1


    for j in range(num_epochs):
        #print('Epoch:', j)
        for data in dl:
            optimizer.zero_grad()
            data=Variable(data)
            output = net(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()


            #print('Loss:', loss.data[0])
        #print()
        #print('############################')

    return net


def autoencoder_dim_reduction(train_X,test_X):
    net=autoencoder_setup(train_X)
    train_X=net.encode(Variable(torch.Tensor(train_X)))
    test_X=net.encode(Variable(torch.Tensor(test_X)))

    return train_X,test_X