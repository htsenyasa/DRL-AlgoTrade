import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F


class FCFFN(nn.Module):
    def __init__(self, inputLayerSize, hiddenLayerSize, outputLayerSize, dropout):
        
        super(FCFFN, self).__init__()
        
        self.fc1 = nn.Linear(inputLayerSize , hiddenLayerSize) 
        self.fc2 = nn.Linear(hiddenLayerSize, hiddenLayerSize) 
        self.fc3 = nn.Linear(hiddenLayerSize, hiddenLayerSize) 
        self.fc4 = nn.Linear(hiddenLayerSize, hiddenLayerSize) 
        self.fc5 = nn.Linear(hiddenLayerSize, outputLayerSize) 

        self.bn1 = nn.BatchNorm1d(hiddenLayerSize)
        self.bn2 = nn.BatchNorm1d(hiddenLayerSize)
        self.bn3 = nn.BatchNorm1d(hiddenLayerSize)
        self.bn4 = nn.BatchNorm1d(hiddenLayerSize)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        # Xavier initialization for the entire neural network
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.xavier_uniform_(self.fc4.weight)
        torch.nn.init.xavier_uniform_(self.fc5.weight)


    def forward(self, input):
        x = self.dropout1(F.leaky_relu(self.bn1(self.fc1(input))))
        x = self.dropout2(F.leaky_relu(self.bn2(self.fc2(x))))
        x = self.dropout3(F.leaky_relu(self.bn3(self.fc3(x))))
        x = self.dropout4(F.leaky_relu(self.bn4(self.fc4(x))))
        output = self.fc5(x)

        return output
