import os
# import xarray
import scipy.io
import torch
from torch.utils.data import Dataset,DataLoader,TensorDataset

datapath__ = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'datasets'))

# matdata1 = scipy.io.loadmat('up.mat')
matdata2 = scipy.io.loadmat(os.path.join(datapath__,'van.mat'))
# upwind=torch.tensor(matdata1['out'])
vanleer=torch.tensor(matdata2['out'])
# upwind.squeeze_(1).squeeze_(-1)
vanler_data = torch.squeeze(vanleer,-1)

class Simulation:
    def __init__(self,array):
      self.dat = array
      self.Length = 32.0
      self.Time = 1.0
      nt,nf,nx = array.shape
      self.nT = nt
      self.nL = nx
      self._caldt()

    def _caldx(self):
      self.dx = self.Length/self.nL
      self.dt = self.Time/self.nT
      self.CFLd = self.dx/self.dt**2
    
    def _caldt(self):
      self.dx = self.Length/self.nL
      self.CFLa = 0.5
      self.dt = self.dx*self.CFLa


def get_loader(x,y,BATCH_SIZE=64):
  tset = TensorDataset(x,y)
  dloader = DataLoader(tset, batch_size=BATCH_SIZE, shuffle=False,drop_last=True)
  return dloader
get_train = lambda x : (x[:-1],x[1:])





# def load_ith_init_condition(i):
#     van_squeeze=sqzfinearray(van_squeeze_all,i)

#     vanleer_ground_coarse=regrid(van_squeeze,8)

#     dat=vanleer_ground_coarse.unsqueeze(1)

#     sim_details = Simulation(dat.shape)

#     num_steps=1
#     get_train = lambda x : (x[:-num_steps],x[num_steps:])
#     x,y = get_train(dat.data) 
#     BATCH_SIZE = x.shape[0]
#     tset = TensorDataset(x,y)
#     dloader = DataLoader(tset, batch_size=BATCH_SIZE, shuffle=False,drop_last=True)
    
#     return dloader,sim_details

