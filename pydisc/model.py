import torch
from torch import nn

# XGRID = 256

"CNN class define"
class AlphaNet(nn.Module):
    def __init__(self,num_filters=16,XGRID=256):
        super().__init__()
        activation = nn.LeakyReLU(0.2)
        self.feat = nn.Sequential(*[
        nn.Conv1d(in_channels=1,out_channels=num_filters,stride=1,
                  kernel_size=3,padding='same',padding_mode='circular'),
        activation,
        nn.MaxPool1d(kernel_size=2,stride=2),
        nn.Conv1d(num_filters,num_filters,stride=1,
                  kernel_size=3,padding='same',padding_mode='circular'),
        activation,
        nn.MaxPool1d(kernel_size=2,stride=2),
        nn.Conv1d(num_filters,num_filters,stride=1,
                  kernel_size=3,padding='same',padding_mode='circular'),
        activation,
        nn.MaxPool1d(kernel_size=2,stride=2),
        nn.Conv1d(num_filters,num_filters,stride=1,
                  kernel_size=3,padding='same',padding_mode='circular')
        ])

        self.fc1=nn.Linear(int(XGRID*num_filters/2**3),4)
#         self.fc2=nn.Linear(120,8)

    def forward(self,x):

        x = self.feat(x)
        x = torch.flatten(x, 1)
        
        alpha = self.fc1(x) ## (bs,4)
        ## polynomial accuracy sum(alpha) = 0
        alphaz = -torch.sum(alpha,1).unsqueeze(1)
        out = torch.concat((alpha,alphaz),1)
        return out #,polynomial

anet=AlphaNet(num_filters=16)