import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(2, 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(2, 2, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        concat = torch.cat([x1, x2], dim=1)
        split1, split2 = torch.chunk(concat, 2, dim=1)
        addition = split1 + split2
        out = self.conv3(addition)
        return out 
    
class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(3, 2, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.conv3 = nn.Conv2d(2, 3, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn = nn.BatchNorm2d(3)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x1 = self.relu(x1)
        x2 = self.conv2(x1)
        x2 = self.relu(x2)
        x3 = self.conv3(x2)
        x3 = self.relu(x3)
        x3 = self.bn(x3)
        return x3 + identity


class ConcatBlock(nn.Module):
    def __init__(self):
        super(ConcatBlock, self).__init__()
        self.conv1 = nn.Conv2d(3, 2, kernel_size=(3, 3), padding=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(3, 2, kernel_size=(3, 3), padding=(1, 1), bias=False)
        self.conv3 = nn.Conv2d(4, 3, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(2)
        self.bn2 = nn.BatchNorm2d(2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.relu(x1)
        x2 = self.conv2(x)
        x2 = self.relu(x2)
        # x2 = self.bn1(x2)
        # x2 = self.bn2(x2)
        x_concat = torch.cat([x1, x2], dim=1)
        x3 = self.conv3(x_concat)
        x3 = self.maxpool(x3)
        return x3


class ModelWithResidualBlocks(nn.Module):
    def __init__(self):
        super(ModelWithResidualBlocks, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=(3, 3), padding=(1, 1), bias=False)
        self.residual_block = ResidualBlock()
        self.concat_block = ConcatBlock()
        self.bn = nn.BatchNorm2d(3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.residual_block(x)
        x = self.concat_block(x)
        return x