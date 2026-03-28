import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self,input_features=7,h1=64,h2=32,h3=16,out=1):
        super().__init__()
        self.fc1=nn.Linear(input_features,h1)
        self.fc2=nn.Linear(h1,h2)
        self.fc3=nn.Linear(h2,h3)
        self.out=nn.Linear(h3,out)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x = self.sigmoid(self.out(x))
        
        return x
    