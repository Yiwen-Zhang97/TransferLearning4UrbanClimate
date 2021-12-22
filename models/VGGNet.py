import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class VGG11Net(nn.Module):
    def __init__(self, input_shape, lr=0.0001):
        super(VGG11Net, self).__init__()
        self.in_channels = input_shape[0]
        self.input_shape = input_shape

        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=(3,3), padding="same")
        self.mp1 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding="same")
        self.mp2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), padding="same")
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding="same")
        self.mp3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), padding="same")
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding="same")
        self.mp4 = nn.MaxPool2d(kernel_size=(2, 2), stride=1)
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding="same")
        self.conv8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding="same")
        self.mp5 = nn.MaxPool2d(kernel_size=(2, 2), stride=1)

        self.flatten_shape = None
        zero_ex = torch.zeros(input_shape).unsqueeze(0)

        with torch.no_grad():
            self.convolutions(zero_ex)

        self.fc1 = nn.Linear(in_features=self.flatten_shape, out_features=128)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(in_features=64, out_features=1)

    def convolutions(self, x):
        x = F.relu(self.conv1(x))
        x = self.mp1(x)
        x = F.relu(self.conv2(x))
        x = self.mp2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.mp3(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.mp4(x)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.mp5(x)

        # Reshape for linear
        x = x.view(x.shape[0], -1)

        if self.flatten_shape is None:
            self.flatten_shape = x.shape[1]

        return x

    def forward(self, x):
        x = self.convolutions(x)

        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = self.fc3(x)

        return x

if __name__ == "__main__":
    model = VGG11Net(input_shape=(3, 33, 33))

