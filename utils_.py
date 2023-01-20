import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

def plot_setting(model, input_, i_sample, output_sample, y_ref):
    
    plt.figure()
    plt.xlim(0, len(input_))
    
    plt.plot(model(input_).detach().numpy(), label='f(x)')
    
    plt.axvline(i_sample, c='grey', ls='--', label='i_sample')

    plt.axhline(output_sample, c='orange', ls='--', label='f(x_sample)')
    plt.axhline(y_ref, c='green', ls='--', label='y_ref')
    plt.axhline(model[-1].bias.detach().numpy(), c='darkred', ls='--', label='last layer bias (min y_ref)')
    plt.arrow(i_sample, output_sample, 
              0, y_ref-output_sample-(y_ref-output_sample)*0.1,
              head_width=3, head_length=abs((y_ref-output_sample)*0.1), 
              fc='k', ec='k', width=1, label='delta y')

    
    plt.xlabel('# input sample')
    plt.ylabel('f(x)')
    plt.legend()
    
    
def rescale_top(model):
    """
    rescale_top : function to equivalent top layers through rescaling (top layer with only +/- 1s)

    Parameters:
    model (torch.nn.Sequential): PyTorch model with the following top-layer-structure: linear-relu-linear.

    Returns:
    torch.nn.Sequential : PyTorch model with equivalent top layers through rescaling.
    """

    w1_new = torch.matmul(torch.diag_embed(torch.abs(model[-1].weight)).squeeze(),model[-3].weight)
    b1_new = model[-3].bias*torch.abs(model[-1].weight.squeeze())
    w2_new = torch.sign(model[-1].weight)

    model_rescaled = model
    
    with torch.no_grad():
        model_rescaled[-3].weight = nn.Parameter(w1_new)
        model_rescaled[-3].bias = nn.Parameter(b1_new)
        model_rescaled[-1].weight = nn.Parameter(w2_new)

    return model_rescaled
