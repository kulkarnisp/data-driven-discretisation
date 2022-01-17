import numpy as np
import matplotlib.pyplot as plt 

import torch
from torch.nn import functional as fn

def conv_pad_circular(u,alpha):
    pad_size=int((alpha.shape[-1])//2)
    padded_out=fn.pad(u,(pad_size,pad_size),mode='circular')
    return fn.conv1d(padded_out,alpha,padding='valid')
     

def predict_one(u,alpha):
    a = alpha.unsqueeze(0).unsqueeze(0)
    u = u.unsqueeze(0)
    return conv_pad_circular(u,a)
    
def predict_series(us,alphas):
    nextu = [predict_one(u,a).squeeze() for u,a in zip(us,alphas)] 
    return torch.stack(nextu).unsqueeze(1)


def plot_compare(xold,xnew,c='br',labels=["Origin","Reconstruct"]):
    avg_err = np.mean((xold-xnew)**2)**0.5
    fig,ax = plt.subplots(ncols=2,figsize=(10,6))
    ax[0].plot(xold,c[0],label=labels[0])
    ax[0].plot(xnew,c[1],label=labels[1])
    ax[0].set_xlabel("Grid index")
    ax[0].set_ylabel("Velocity")
    err = np.mean((xold-xnew)**2)**0.5
    print(f"Average Error is {avg_err:.3e}")
    ax[0].set_title(f"Recon err:{err:.4e}")
    ax[0].legend()
    ax[1].plot(xold-xnew,"g")
    ax[1].set_title(f"Error Comparison")
    ax[1].set_xlabel("Grid index")
    ax[1].set_ylabel("Errors")
    # plt.show()
    return 0
