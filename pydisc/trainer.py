# from ast import Interactive
from ipywidgets import interactive
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from .helper import predict_series as prediction
from .model import AlphaNet
from .loader import DataLoader,get_train,get_loader
from .helper import plot_compare


class Trainer:
    def __init__(self,data,BATCH_SIZE=64,LR=0.005,EPOCHS=40) -> None:
        self.data = data
        self.xtrain,self.ytrain = get_train(data)
        nt,nf,nx = data.shape
        # self.BATCH_SIZE = BATCH_SIZE
        self.LR = LR
        self.EPOCHS = EPOCHS

        self.net = AlphaNet(num_filters=16,XGRID=nx)
        self.loader = get_loader(self.xtrain,self.ytrain,
                                BATCH_SIZE=BATCH_SIZE)

    def fit(self,do_eval=True):
        self.loss_arr = trainn(self.loader,self.net,self.EPOCHS,self.LR)
        if do_eval:
            self.train_eval()
        # self.yhat = self.predict(self.xtrain)

    def predict(self,x):
        yhat =  x- prediction(x,self.net(x)) #*DTIME
        return  yhat.detach()

    def train_eval(self):
        self.net = self.net.eval()
        self.ypred = self.predict(self.xtrain)
        self.train_accuracy = torch.mean((self.ytrain-self.ypred)**2)**0.5

    def test_eval(self,testdata):
        x,y = get_train(testdata)
        self.net = self.net.eval()
        yhat = self.predict(x)
        self.test_accuracy = torch.mean((y-yhat)**2)**0.5
        self.yhat = yhat
        self.ytest = y
        # return 

    def eplot(self,y,yhat):
        y = y.squeeze_().detach().numpy()
        yhat = yhat.squeeze_().detach().numpy()
        fig = plot_compare(y,yhat)
        plt.show()
        # return 0

    def jplot(self,j):
        # self.xtest = 
        self.eplot(self.ytrain[j],self.ypred[j])
        return 0
        # s=0

    def iplot(self,x,y):
        f = lambda j : self.eplot(x[j],y[j])
        return interactive(f,j=range(len(x)))
       
    def time_stepping(self,tend,xinit):
        y = xinit.reshape(1,1,-1)
        outarr = []
        for i in range(int(tend)):
            y = self.predict(y)
            outarr.append(y[0].data.numpy()) 
        return torch.Tensor(np.array(outarr))

    

def trainn(datloader,anet,N_Epochs=40,LR=0.005):

    iscuda = torch.cuda.is_available()
    if iscuda:
        anet = anet.cuda()
    optimizer = torch.optim.Adam(anet.parameters(),lr=LR)
    criterion = nn.MSELoss()

    loss_ls=[]
    for epoch in tqdm(range(N_Epochs)):
        for ii,(x,y) in enumerate(datloader):
            if iscuda:
                x = x.cuda()
                y = y.cuda()   
            optimizer.zero_grad()
            alphav=anet(x)


            upre = x.data
            yhat = upre - prediction(upre,alphav) #*DTIME #*DTime
            
            loss = criterion(yhat,y)
            loss.backward()
            optimizer.step()
            loss_ls.append(loss.item())
    #         print(epoch)
    #     loss_ls=torch.tensor(loss_ls)
    plt.plot(loss_ls)
    plt.title("Learning curve")
    plt.show()
    #     print(loss_ls)
    return loss_ls
#     return cnn

