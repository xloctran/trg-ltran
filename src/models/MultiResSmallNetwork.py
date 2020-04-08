# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import torch
from torch import nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class SmallNetwork(nn.Module):
    def __init__(self, nclasses=2 , outputsize = 8):
        super(SmallNetwork,self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=3),
            nn.ReLU(True),
            nn.BatchNorm2d(6),
            nn.Conv2d(6,8,stride=2,kernel_size=3),
            nn.ReLU(True),
            nn.BatchNorm2d(8),
            nn.Conv2d(8,8,stride=2,kernel_size=3),
            nn.ReLU(True),
            nn.BatchNorm2d(8),
            nn.Conv2d(8,8,stride=2,kernel_size=3),
            nn.ReLU(True),
            nn.BatchNorm2d(8),
            nn.Conv2d(8,8,stride=2,kernel_size=3),
            nn.ReLU(True),
            nn.BatchNorm2d(8),
            nn.Conv2d(8,8,stride=1,kernel_size=2),
            nn.ReLU(True),
            nn.BatchNorm2d(8) )
        
        self.fc= nn.Sequential(
            Flatten(),
            nn.Linear(8, outputsize),
            nn.ReLU(True),
            nn.BatchNorm1d(outputsize),
            nn.Linear(outputsize, outputsize),
            nn.ReLU(True),
            nn.BatchNorm1d(outputsize) )
        
        self.classifier=  nn.Sequential( nn.Linear(outputsize, nclasses))

        
    def forwardlin(self,x):
        x = self.encoder(x)
        return self.fc(x)
        
    def forward(self,x):
        x = self.encoder(x)
        x = self.fc(x)
        return self.classifier(x)

# -

class MergeSmallNetworks(nn.Module):
    def __init__(self,nclasses=2, outputsize = 8 ):
        super(MergeSmallNetworks,self).__init__()
        
        self.res1 = SmallNetwork(outputsize = outputsize, nclasses= nclasses)
        self.res2 = SmallNetwork(outputsize = outputsize, nclasses= nclasses)
        
        self.final= nn.Sequential(
            nn.Linear(outputsize * 2, 16),
            nn.ReLU(True),
            nn.BatchNorm1d(16),
            nn.Linear(16, nclasses))
        

    def dualfoward(self,xres1,xres2):
        res1 = self.res1.forwardlin(xres1)
        res2 = self.res2.forwardlin(xres2)
        res1res2 = torch.cat([res1, res2], 1)
        return self.final(res1res2)

