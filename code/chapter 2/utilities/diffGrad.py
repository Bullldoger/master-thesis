import pickle
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import autograd as grad

import math
from torch.optim.optimizer import Optimizer
import numpy as np
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DiffGrad(Optimizer):
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, version=0, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        
        super().__init__(params, defaults)
        self.version = version

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure=None):
        
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('diffGrad does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['previous_grad'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq, previous_grad = state['exp_avg'], state['exp_avg_sq'], state['previous_grad']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                if self.version==0:
                    diff = abs(previous_grad - grad)
                elif self.version ==1:
                    diff = previous_grad-grad
                elif self.version ==2:
                    diff =  .5*abs(previous_grad - grad)
                    
                if self.version==0 or self.version==1:    
                    dfc = 1. / (1. + torch.exp(-diff))
                elif self.version==2:
                    dfc = 9. / (1. + torch.exp(-diff))-4
                    
                state['previous_grad'] = grad
                exp_avg1 = exp_avg * dfc
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                p.data.addcdiv_(-step_size, exp_avg1, denom)
        return loss