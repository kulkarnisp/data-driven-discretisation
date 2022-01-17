
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

    def fit(self):
        self.loss_arr = trainn(self.loader,self.net,self.EPOCHS,self.LR)
        self.yhat = self.net(self.xtrain)

    def predict(self,x):
        yhat =  x- prediction(x,self.net(x)) #*DTIME
        return  yhat.detach()

    def eplot(self,y,yhat):
        y = y.squeeze_().detach().numpy()
        y = yhat.squeeze_().detach().numpy()
        fig = plot_compare(y,yhat)
        plt.show()
        return 0

    def eval(self,test_dat):
        x,y = get_train(test_dat)
        # self.xtest = 
        yhat = self.predict(x)
        return y,yhat
        s=0

    # def     
    

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
