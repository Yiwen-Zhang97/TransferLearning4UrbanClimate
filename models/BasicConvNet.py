import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
class ConvNet(nn.Module):
    def __init__(self, input_shape, lr=0.0001):
        super(ConvNet, self).__init__()
        self.in_channels = input_shape[0]
        self.input_shape = input_shape

        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=16, kernel_size=(3,3), padding="same")
        self.conv1_bn = nn.BatchNorm2d(16)
        self.mp1 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), padding="same")
        self.conv2_bn = nn.BatchNorm2d(32)
        self.mp2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding="same")
        self.conv3_bn = nn.BatchNorm2d(64)
        self.mp3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.flatten_shape = None
        zero_ex = torch.zeros(input_shape).unsqueeze(0)

        with torch.no_grad():
            self.convolutions(zero_ex)

        self.fc1 = nn.Linear(in_features=self.flatten_shape, out_features=1)
        self.drop1 = nn.Dropout(0.15)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        # self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(in_features=32, out_features=1)

    def convolutions(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = self.mp1(x)
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = self.mp2(x)
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = self.mp3(x)

        # Reshape for linear
        x = x.view(x.shape[0], -1)

        if self.flatten_shape is None:
            self.flatten_shape = x.shape[1]

        return x

    def forward(self, x):
        x = self.convolutions(x)
        x = self.fc1(x)
        # # x = self.drop1(x)
        # x = F.elu(self.fc2(x))
        # x = self.fc3(x)
        return x