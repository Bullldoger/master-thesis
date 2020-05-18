import torch
import torch.nn as nn
import time
import numpy as np


class CustomActivation(nn.Module):
    def __init__(self, activation = None):
        super().__init__()
        self.activation = activation
        
    def forward(self, x):
        return self.activation(x)

    
class FourierTerm(nn.Module):
    def __init__(self, input_dim = 2):
        super().__init__()
        self.input_dim = input_dim
        self.activation_cos = torch.cos
        self.activation_sin = torch.sin
        self.bias = nn.Parameter(data=torch.tensor([[0.0]]))
        self.output_sin = nn.Linear(input_dim, 1)
        self.output_cos = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        sin = self.output_sin(self.activation_sin(x))
        cos = self.output_cos(self.activation_cos(x))

        return sin + cos + self.bias
    
    
class FourierFCN2D(nn.Module):
    def __init__(self, layer_size = 8, save_folder='trained_models'):
        super().__init__()
        
        self.layer_size = layer_size
        self.path = save_folder
        self.u = FourierTerm(input_dim=layer_size)
        self.v = FourierTerm(input_dim=layer_size)
        self.pi= np.pi
        
    def prepaire(self, x):
        
        X = torch.cat([
            (self.pi * x * l)[:, None] for l in range(1, self.layer_size + 1)
        ], 1)
        
        return X
        
    def forward(self, x):
        
        X = self.prepaire(x[:, 0])
        Y = self.prepaire(x[:, 1])
        R = X + Y
        
        U = self.u(X)
        V = self.v(X)
        
        output = torch.cat((U, V), 1)
        return output
    
    def save_model(self):
        torch.save(self.state_dict(), '{}/{}.pt'.format(self.path, self.name))


