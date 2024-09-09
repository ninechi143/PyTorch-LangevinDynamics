# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os

import torch
import torch.nn as nn


"""
PyTorch implementation of Langevin Dynamics
Here we consider a probability, P(x) = P_hat(x) / Z, 
where x is in [-0.7, 1.5], P_hat(x) = (x-0.4)^4 - (x-0.2)^2 + 0.6, and Z is normalization constant
to this end, Z = integral from -0.7~1.5 of P_hat(x), and by calculation we can know that Z is 0.98887
"""

class Energy_function(nn.Module):

    def __init__(self):
        super(Energy_function, self).__init__()
           
    def forward(self, x):
        p_hat = (x - 0.4)**4 - (x-0.2)**2 + 0.6
        energy = -1 * torch.log(p_hat)
        return p_hat


def make_gif():

    from pathlib import Path
    import cv2
    import imageio

    png_list = list(Path(os.path.join(os.path.dirname(__file__))).rglob("*.png"))
    png_list.sort()
    process = [cv2.imread(str(i))[..., ::-1] for i in png_list]
    imageio.mimsave(os.path.join(os.path.dirname(__file__), "langevin_dynamics.GIF") , process , duration = 1000/30)
    [os.remove(i) for i in png_list]


# def parse_args():
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--note', default = "1", choices=('1', '2', '3'))
#     parser.add_argument('--stepsize', type=float, default=0.01, help='Langevin dynamics step size. default 0.01')

#     args = parser.parse_args()
#     return args

def main():

    # args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Energy_function().to(device)
    

    n_epoch = 500
    n_datapoint = 5000

    SGLD_step = 0.01
    SGLD_noise_scale = np.sqrt(2 * SGLD_step)
    SGLD_decay = np.power(0.001, 1 / n_epoch) # for annealed SGLD

    plot_x = torch.arange(-0.7, 1.5, (1.5+0.7)/1000).to(device)
    plot_y = model(plot_x) / 0.98887 # divided by normalization constant

    plot_x = plot_x.detach().cpu().numpy()
    plot_y = plot_y.detach().cpu().numpy()
    
    X = (torch.zeros(n_datapoint) + 0.4).to(device)
    # X = (torch.randn(n_datapoint) + 0.4).to(device)
    for i in tqdm(range(n_epoch)):
        
        #----------------------------------------------------------------------------------------
        # SGLD start
        X.requires_grad = True

        log_p_hat = torch.log(model(X))
        grad = torch.autograd.grad(log_p_hat.sum(), X, only_inputs=True, retain_graph=True)[0]

        X = X + SGLD_step * grad + SGLD_noise_scale * torch.randn_like(X)
        X = torch.clamp(X, min = -0.7, max = 1.5)
        X = X.detach()

        SGLD_step = SGLD_step * SGLD_decay # for annealed SGLD
        SGLD_noise_scale = np.sqrt(2 * SGLD_step)
        # SGLD end
        #----------------------------------------------------------------------------------------
        # for visualization
        data = X.cpu().numpy()
        hist, bin_edges = np.histogram(data.reshape([-1]), bins = 100, range = (-0.7, 1.5), density = True)
        
        plt.bar(bin_edges[:-1], hist, width = (2.2)/100)
        plt.plot(plot_x, plot_y, c = "red")

        plt.title("Langevin Dynamics")
        plt.xlabel("x")
        plt.ylabel("P(x)")

        plt.xlim(-1, 1.8)
        plt.ylim(0, 2.)
        plt.savefig(os.path.join(os.path.dirname(__file__), f"{i:04d}.png"))
        # plt.show()
        plt.close("all")
    
    make_gif()
    return

if __name__ == "__main__":
    
    main()
