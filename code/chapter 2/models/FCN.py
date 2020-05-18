import torch
import torch.nn as nn
import time

class CustomActivation(nn.Module):
    def __init__(self, activation = None):
        super().__init__()
        self.activation = activation
        
    def forward(self, x):
        return self.activation(x)

def generate_sequence(input_dim = 2, layer_size = 8, layers_num = 1, output_dim = 1):
    
    sequence = nn.Sequential()
    sequence.add_module('Linear layer-[{}]: {}->{}'.format(0, input_dim, layer_size), nn.Linear(input_dim, layer_size))
    sequence.add_module('Activation function[{}]'.format(0), nn.Tanh())
    for l_num in range(layers_num):
        sequence.add_module('Linear layer-[{}]: {}->{}'.format(l_num + 1, layer_size, layer_size), nn.Linear(layer_size, layer_size))
        sequence.add_module('Activation function[{}]'.format(l_num + 1), nn.Tanh())
    sequence.add_module('Linear layer-[{}]: {}->{}'.format(layers_num + 1, layer_size, output_dim), nn.Linear(layer_size, output_dim))
    return sequence


class FCN(nn.Module):
    def __init__(self, input_dim = 2, layer_size = 8, layers_num = 1, output_dim = 1, save_folder='trained_models'):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim= output_dim
        self.layer_size= layer_size
        self.layers_num= layers_num
        
        self.path = save_folder
        self.name = 'FCN:{}:[{}:{}]:{}:{}'.format(input_dim, layer_size, layers_num, output_dim, time.time_ns())
        self.model = self._construct()
        
    def _construct(self):
        model = generate_sequence(input_dim = self.input_dim, layer_size = self.layer_size, layers_num = self.layers_num, output_dim = self.output_dim)
        return model
    
    def forward(self, x):
        return self.model(x)
    
    def save_model(self):
        torch.save(self.state_dict(), '{}/{}.pt'.format(self.path, self.name))
        return self
    
class FCNCustom(nn.Module):
    def __init__(self, input_dim = 2, layer_size = 8, layers_num = 1, output_dim = 1, save_folder='trained_models', activate=lambda x: torch.cos(x) - x):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim= output_dim
        self.layer_size= layer_size
        
        self.path = save_folder
        self.name = 'FCN:{}:[{}:{}]:{}:{}'.format(input_dim, layer_size, layers_num, output_dim, time.time_ns())
        
        self.l1 = nn.Linear(input_dim, layer_size)
        self.l2 = nn.Linear(layer_size, output_dim)
        self.activate = CustomActivation(activation=activate)
    
    def forward(self, x):
        return self.l2(self.activate(self.l1(x)))
    
    def save_model(self):
        torch.save(self.state_dict(), '{}/{}.pt'.format(self.path, self.name))
        return self
    
class FCNFourier(FCNCustom):
    def __init__(self, input_dim = 2, layer_size = 8, layers_num = 1, output_dim = 1, save_folder='trained_models'):
        super().__init__(input_dim = 2, layer_size = 8, layers_num = 1, output_dim = 1, save_folder='trained_models', activate=lambda x: torch.cos(x))
    