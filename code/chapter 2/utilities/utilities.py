import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import autograd as grad
import math
from torch.optim.optimizer import Optimizer
import numpy as np
import torch.nn as nn
import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
step = 1e-2

def analytical_gradients(oracle, arguments, function=0, variables=(0, ), **kwargs):
    variable = variables[0]
    function_values = oracle(arguments)[:, variable]
    
    derivative = grad.grad(function_values.sum(), arguments, create_graph=True)[0][:, variable]
    for variable in variables[1:]:
        derivative = grad.grad(derivative.sum(), arguments, create_graph=True)[0][:, variable]
    
    return derivative

def numerical_gradients(oracle, arguments, function=0, variables=(0, ), **kwargs):
    derivative = None
    variables_dimension = kwargs.get('variables_dimension', 1)
    function_dimension  = kwargs.get('function_dimension', 1)
    
    if len(variables) == 0:
        derivative = oracle(arguments)
    elif len(variables) == 1:
        d = torch.zeros(1, variables_dimension).to(device)
        (_v1, ) = variables
        d[0, _v1] = step
        derivative = (oracle(arguments + d) - oracle(arguments - d)) / (2 * step)
#         print(derivative.shape)
    elif len(variables) == 2:
        d1 = torch.zeros(1, variables_dimension).to(device)
        d2 = torch.zeros(1, variables_dimension).to(device)
        _v1, _v2 = variables
        d1[0, _v1] = step
        d2[0, _v2] = step
        derivative = (oracle(arguments + d1 + d2) - oracle(arguments + d1 - d2) \
                      - oracle(arguments + d2 - d1) + oracle(arguments - d1 - d2)) / (4 * step ** 2)
#         print(derivative.shape)
    else:
        return
    
    return derivative
    
def Gradients(oracle, arguments, function=0, variables=(0, ), **kwargs):
    as_numpy = kwargs.get('as_numpy', False)
    meta_data= kwargs.get('meta_data', False)
    numerical= kwargs.get('numerical', True)
    
    function_values = oracle(arguments)
    variables_dimension = arguments.size()[1]
    function_dimension  = function_values.size()[1]
    
    gradients_method = numerical_gradients if numerical else analytical_gradients
    gradient = gradients_method(oracle, arguments, function=function, 
                                variables=variables, variables_dimension=variables_dimension,
                                function_dimension=function_dimension)[:, function][:, None]
    
    if as_numpy:
        gradient = gradient.cpu().detach().numpy()
        function_values = function_values.cpu().detach().numpy()
        
    if meta_data:
        return gradient, function_values
    
    return gradient

def Equation(oracle, arguments, terms=[(0, 0), (1, 1), (0, 1)], phi=lambda x, y, z: x + y + z, numerical=True):
    phi_arguments = []
    for _equation in terms:
        function, *variables = _equation
        phi_arguments += [Gradients(oracle, arguments, function=function, numerical=numerical, variables=variables)]
    return phi(*phi_arguments)

def training(oracle, equations=dict(), **kwargs):
    
    LOSS = lambda x: torch.mean(x ** 2)
    LOSS = kwargs.get('loss', LOSS)
    
    if 'parameters' in kwargs:
        parameters = kwargs['parameters']
    else:
        parameters = oracle.parameters()
    optimizer = kwargs.get('optimizer', torch.optim.Adam)
    epochs = kwargs.get('epochs', 1500)
    opt = optimizer(parameters, lr=1e-2)
    regularization = kwargs.get('regularization', lambda x: 0.0)
    
    training_errors = list()
    
    conditions_errors = dict()
    for error_name in equations.keys():
        conditions_errors[error_name] = list()
    
    for epoch in tqdm.tqdm(range(epochs + 1)):
        opt.zero_grad()
        regularizator = regularization(oracle)
        loss = regularizator
        for equation_name, equation in equations.items():
            beta = equation.get('regularization', 1.0)
            points = equation['points']
            function_values = oracle(points)
            nt = Equation(oracle, points, terms=equation['derivatives'], phi=equation['phi']).to(device)
            right_hand_side = equation['psi'](points).to(device)
            equation_loss = beta * LOSS(nt - right_hand_side)
            loss += equation_loss
            conditions_errors[equation_name] += [equation_loss.item()]
        loss.backward()
        opt.step()
        training_errors += [loss.item()]
    return training_errors, conditions_errors

def get_facets_pts(facets, pts, *args, **kwargs):
    inds = np.apply_along_axis(facets[0].inside, 1, pts, True)
    for facet in facets[1:]:
        inds = inds | np.apply_along_axis(facet.inside, 1, pts, *args, **kwargs)
    inds = np.nonzero(inds.astype(np.int))[0]
    return pts[inds]

def get_points(fixed={}, size=25, dim=2):
    fixed_arg = set([i for i, _ in fixed.items()])
    result = []
    for arg in range(dim):
        if arg in fixed_arg:
            if not isinstance(fixed[arg], list):
                result += [torch.tensor([fixed[arg]], requires_grad=True)]
            else:
                result += [torch.tensor(fixed[arg], requires_grad=True)]
        else:
            result += [torch.linspace(0, 1.0, size, requires_grad=True)]
            
    vls = [_.flatten() for _ in torch.meshgrid(*result)]
    result = torch.stack(vls).T.to(device)
    return result

def get_points_randomly(geometry, generated_size, boundary=True):
    size = 10000
    _generator = geometry.random_points
    if boundary:
        _generator = geometry.random_boundary_points
    pts = torch.tensor(_generator(size), dtype=torch.float).to(device)
    generator = lambda : pts[torch.randperm(pts.shape[0])[:generated_size]]
    return generator

def plot_error(errors, epochs=500, figsize=(16, 10)):
    
    errors = np.array(errors)
    iters = np.array(list(range(epochs + 1)))

    plt.figure(figsize=figsize)
    plt.title('Learning results', fontsize=24)
    plt.semilogy(iters, errors, 'g', linewidth=1, label=r'$\mathcal{L}$')
    plt.xlabel('#Epoch', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.legend(fontsize=14)
    plt.grid()
    
    
def plot_error_info(info, epochs=500):
    plt.figure(figsize=(16, 10))
    plt.title('Learning results', fontsize=24)
    iters = np.array(list(range(epochs + 1)))

    for error_key in info.keys():
        error = np.log10(info[error_key])
        plt.plot(iters, error, label=error_key)

    plt.xlabel('#Epoch', fontsize=18)
    plt.ylabel('log(Loss)', fontsize=18)
    plt.legend(fontsize=14)
    plt.grid()
    plt.show()
    