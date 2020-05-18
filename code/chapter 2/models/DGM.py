class SelfDGM(nn.Module):
    def __init__(self, n_layers, n_nodes, dimensions=1, activation=nn.Tanh):
        super().__init__()
        
        self.d1 = DGMNet(n_layers, n_nodes, dimensions=3, output=1, activation=nn.Tanh)
        self.d2 = DGMNet(n_layers, n_nodes, dimensions=3, output=1, activation=nn.Tanh)
        self.d3 = DGMNet(n_layers, n_nodes, dimensions=3, output=1, activation=nn.Tanh)
        
    def forward(self, X):
        
        S1 = self.d1(X)
        S2 = self.d2(X)
        S3 = self.d3(X)
        
        S = torch.cat([S1, S2, S3], 1)
        
        return S

class DGMNet(nn.Module):
    def __init__(self, n_layers, n_nodes, dimensions=1, output=1, activation=nn.Tanh):
        """
        Parameters:
            - n_layers:     number of layers
            - n_nodes:      number of nodes in (inner) layers
            - dimensions:   number of spacial dimensions
        """
        super().__init__()
        
        self.n_layers = n_layers
        self.initial_layer = DenseLayer(dimensions, n_nodes, activation=activation())

        self.lstm_1 = LSTMLikeLayer(dimensions, n_nodes, activation=activation())
        self.lstm_2 = LSTMLikeLayer(dimensions, n_nodes, activation=activation())
        self.lstm_3 = LSTMLikeLayer(dimensions, n_nodes, activation=activation())
        self.lstm_4 = LSTMLikeLayer(dimensions, n_nodes, activation=activation())
        self.lstm_5 = LSTMLikeLayer(dimensions, n_nodes, activation=activation())
        
        self.final_layer = DenseLayer(n_nodes, output, activation=lambda x: x)


    def forward(self, X):
        
        S = self.initial_layer(X)
        
        S = self.lstm_1(S, X)
        S = self.lstm_2(S, X)
        S = self.lstm_3(S, X)
        S = self.lstm_4(S, X)
        S = self.lstm_5(S, X)
        result = self.final_layer(S)

        return result
    

class DenseLayer(nn.Module):
    def __init__(self, n_inputs, n_outputs, activation):
        """
        Parameters:
            - n_inputs:     number of inputs
            - n_outputs:    number of outputs
            - activation:   activation function
        """
        super().__init__()

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        
        self.W = nn.Linear(self.n_inputs, self.n_outputs)
        self.f = nn.Tanh()
    
    
    def forward(self, inputs):
        h = self.f(self.W(inputs))
        return h



class LSTMLikeLayer(nn.Module):
    def __init__(self, n_inputs, n_outputs, activation):
        """
        Parameters:
            - n_inputs:     number of inputs
            - n_outputs:    number of outputs
            - activation:   activation function
        """
        super().__init__()

        self.n_outputs = n_outputs
        self.n_inputs = n_inputs
        
        self.Uz = nn.Linear(self.n_inputs, self.n_outputs)
        self.Ug = nn.Linear(self.n_inputs, self.n_outputs)
        self.Ur = nn.Linear(self.n_inputs, self.n_outputs)
        self.Uh = nn.Linear(self.n_inputs, self.n_outputs)
        self.Wz = nn.Linear(self.n_outputs, self.n_outputs)
        self.Wg = nn.Linear(self.n_outputs, self.n_outputs)
        self.Wr = nn.Linear(self.n_outputs, self.n_outputs)
        self.Wh = nn.Linear(self.n_outputs, self.n_outputs)
        
        self.f = activation

    
    def forward(self, S, X):
        
        Z = self.f(self.Uz(X) + self.Wz(S))
        G = self.f(self.Ug(X) + self.Wg(S))
        R = self.f(self.Ur(X) + self.Wr(S))
        H = self.f(self.Uh(X) + self.Wh(S * R))
        
        Z_= (torch.ones_like(G) - G) * H + Z * S
        
        return Z_