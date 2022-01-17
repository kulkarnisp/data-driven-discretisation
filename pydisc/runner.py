import imp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from ipywidgets import interact,interactive

import torch
import torch.nn as nn

from .loader import vanler_data,get_train
from .loader import get_loader,Simulation
from .model import AlphaNet
from .helper import plot_compare
from .helper import predict_series as prediction
from .trainer import trainn

# van
# SAMPLING = 4
def fit(sim_details):
    # sim_data = vanler_data[::1]

    return anet,loss_arr


class RunModel:
    def __init__(data,self) -> None:    
        # sim_details = Simulation(sim_data)
        self.details = Simulation(data)
        self.dat = data
        nt,nf,nx = data.shape
        dat_set = get_train(data)
        self.loader = get_loader(*dat_set,BATCH_SIZE=64)

        anet = AlphaNet(num_filters=16,XGRID=nx)
        self.fit()

    def fit(self):
        self.loss_arr = trainn(sim_details,anet)
        # pass

def feval(j):
    x,y,yhat = train_inference(j)



def train_inference(dat,enet,j=2):
    x,y = dat[j-1:j],dat[j:j+1]
    yhat = x- prediction(x,enet(x)) #*DTIME
    
    yhat = yhat.squeeze_().detach().numpy()



def run(sim_details,anet):
    enet = anet.cpu().eval()

    # sim_data = vanler_data[::SAMPLING]
    dat = sim_details.dat.cpu()
    DTIME = sim_details.dt
    # j = 150

    return interactive(feval,j=range(1,len(dat)-1))
