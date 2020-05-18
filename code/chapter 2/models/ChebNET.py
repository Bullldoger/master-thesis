import torch
import torch.nn as nn
import time

class ChebFCN2D(nn.Module):
    def __init__(self, layer_size = 8, output_dim=3, save_folder='trained_models', **kwargs):
        super().__init__()
        
        self.layer_size = layer_size
        self.path = save_folder
        self.output_dim = output_dim
        self.linear = nn.Linear((layer_size - 1) * (layer_size - 1) + 1, self.output_dim)
        
        
    def prepaire(self, x):
        x_pp = torch.ones_like(x)
        x_p  = torch.ones_like(x) * x
        X = torch.cat((x_pp, x_p), 1)
        for l in range(2, layer_size):
            xc = 2 * x * x_p - x_pp
            X = torch.cat((X, xc), 1)
            x_pp = x_p
            x_p  = xc
        return X
    
    def product(self, X, Y):
        
        R = torch.ones_like(X[:, 0][:, None])
        for i in range(1, self.layer_size):
            for j in range(1, self.layer_size):
                R = torch.cat((R, self._product(X[:, i], Y[:, j])), 1)
        return R
    
    def _product(self, x, y):
        return (x * y)[:, None]
        
    def forward(self, x):
        
        X = self.prepaire(x[:, 0][:, None])
        Y = self.prepaire(x[:, 1][:, None])
        
        R = self.product(X, Y)
        output = self.linear(R)
        
        return output
    
    def save_model(self):
        torch.save(self.state_dict(), '{}/{}.pt'.format(self.path, self.name))
        

class ChebFCN1D(nn.Module):
    def __init__(self, layer_size = 8, output_dim=3, save_folder='trained_models', **kwargs):
        super().__init__()
        
        self.layer_size = layer_size
        self.path = save_folder
        self.output_dim = output_dim
        self.linear = nn.Linear(layer_size, self.output_dim)
        
    def prepaire(self, x):
        x_pp = torch.ones_like(x)
        x_p  = torch.ones_like(x) * x
        X = torch.cat((x_pp, x_p), 1)
        for l in range(2, self.layer_size):
            xc = 2 * x * x_p - x_pp
            X = torch.cat((X, xc), 1)
            x_pp = x_p
            x_p  = xc
        return X
    
    def forward(self, x):
        
        X = self.prepaire(x[:, 0][:, None])
        output = self.linear(X)
        
        return output
    
    def save_model(self):
        torch.save(self.state_dict(), '{}/{}.pt'.format(self.path, self.name))