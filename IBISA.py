import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import PIL.ImageOps
from torchvision import transforms
from utils import *

random.seed(32984)
torch.manual_seed(2234)

class InfoModel(nn.Module):
    def __init__(self, core, bottleneck):
        super(InfoModel, self).__init__()

        self.core = core
        for parameter in self.core.parameters():
            parameter.requires_grad = False

        self.botteneck = bottleneck

    def forward(self, x):
        x = self.botteneck(x)
        y = self.core(x)
        return y

    def get_saliency(self):
        return self.botteneck.get_lamb()

    def reset_model(self):
        self.botteneck.reset_alpha()

    def activations_hook(self, module, inputs, act):
        self.activations = act

    def get_activations(self):
        return self.activations.detach()

class InfoLayer(nn.Module):
    def __init__(self, input_size, mask_size, mask_range):
        super(InfoLayer, self).__init__()

        self.input_size = input_size
        self.mask_size = mask_size
        self.mask_range = mask_range

        self.alpha = nn.Parameter(torch.empty((self.mask_size, self.mask_size), dtype=torch.float32))
        self.reset_alpha()

    def forward(self, x):
        alpha_ex = self.alpha.expand((1, 1, self.mask_size, self.mask_size))
        alpha_ex = nn.Upsample(size=(self.input_size, self.input_size), mode='bicubic')(alpha_ex)
        # alpha_ex = nn.Upsample(size=(self.input_size, self.input_size), mode='bilinear')(alpha_ex)
        self.lamb = torch.sigmoid(alpha_ex)

        return torch.mul(x, self.lamb)

    def get_lamb(self):
        return self.lamb

    def reset_alpha(self):
        nn.init.uniform_(self.alpha, -self.mask_range, self.mask_range)

class InfoLoss(nn.Module):
    def __init__(self, beta, phi):
        super(InfoLoss, self).__init__()
        self.beta = beta
        self.phi = phi

    def forward(self, saliency_map):
        sum_saliency = torch.sum(torch.abs(saliency_map))
        complexity_loss = torch.mul(self.beta, sum_saliency)
        variation_loss = total_variation(saliency=saliency_map, phi=self.phi)

        return complexity_loss + variation_loss

def total_variation(saliency, phi):
    saliency = torch.squeeze(saliency)
    x_diff = torch.abs(saliency[1:,:] - saliency[:-1,:])
    y_diff = torch.abs(saliency[:,1:] - saliency[:,:-1])

    return torch.mul(torch.sum(x_diff) + torch.sum(y_diff), phi)

def train_bottleneck(model, x, y, epochs, loss_ce, loss_inf, sigma, opt):
    for i in range(epochs):
        opt.zero_grad()        
        y_pred = model(x)
        loss = sigma*loss_ce(y_pred, y) + loss_inf(model.get_saliency())
        loss.backward()
        opt.step()
    
    return model.get_saliency()
    

def gaussian(mu, std, n_points=10000):
    x = np.linspace(start=-5*std, stop=5*std, num=n_points)

    cte_term = 1/np.sqrt(2*np.pi*std**2)
    exp_term = ((x-mu)**2)/(2*std**2)

    return x, cte_term*np.exp(-exp_term)