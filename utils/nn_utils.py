import math
import sklearn
import torch 
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.nn.functional as F
import sklearn.metrics

class fcbr(nn.Module):
    """ fc-bn-relu
    [B, Fin] -> [B, Fout]
    """
    def __init__(self, Fin, Fout):
        super(fcbr, self).__init__()
        self.fc = nn.Linear(Fin, Fout)
        self.bn = nn.BatchNorm1d(Fout)
        self.ac = nn.ReLU(True)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.ac(x)
        return x

class fcdbr(nn.Module):
    """ fc-dp-bn-relu
    [B, Fin] -> [B, Fout]
    """
    def __init__(self, Fin, Fout, dp=0.5):
        super(fcdbr, self).__init__()
        self.fc = nn.Linear(Fin, Fout)
        self.dp = nn.Dropout(dp)
        self.bn = nn.BatchNorm1d(Fout)
        self.ac = nn.ReLU(True)

    def forward(self, x):
        x = self.fc(x)
        x = self.dp(x)
        x = self.bn(x)
        x = self.ac(x)
        return x

class conv1dbr(nn.Module):
    """ Conv1d-bn-relu
    [B, Fin, N] -> [B, Fout, N]
    """
    def __init__(self, Fin, Fout, kernel_size):
        super(conv1dbr, self).__init__()
        self.conv = nn.Conv1d(Fin, Fout, kernel_size)
        self.bn = nn.BatchNorm1d(Fout)
        self.ac = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x) # [B, Fout, N]
        x = self.bn(x)
        x = self.ac(x)
        return x

class conv2dbr(nn.Module):
    """ Conv2d-bn-relu
    [B, Fin, H, W] -> [B, Fout, H, W]
    """
    def __init__(self, Fin, Fout, kernel_size, stride=1):
        super(conv2dbr, self).__init__()
        self.conv = nn.Conv2d(Fin, Fout, kernel_size, stride)
        self.bn = nn.BatchNorm2d(Fout)
        self.ac = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x) # [B, Fout, H, W]
        x = self.bn(x)
        x = self.ac(x)
        return x
