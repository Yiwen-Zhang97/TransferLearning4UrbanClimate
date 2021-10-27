import sys
sys.path.append("models")
sys.path.append("utils")

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from BasicConvNet import ConvNet
from VGGNet import VGG11Net
from torch_dataloader import LSTDataLoader

train_dataloader, test_dataloader = LSTDataLoader("data/images.npy", "data/labels.npy").build_loaders()

EPOCHS = 500
LEARNING_RATE = 5e-6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = ConvNet(input_shape=(3, 33, 33)).to(DEVICE)

loss_fn = nn.L1Loss()
optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE)

test_loss = []
train_loss = []
for epoch in range(EPOCHS):
    running_train_loss = 0
    train_n_iter = 0
    running_test_loss = 0
    test_n_iter = 0

    model.train()
    for images, labels in tqdm(train_dataloader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        pred = model(images)
        loss = loss_fn(pred, labels)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()
        train_n_iter += 1

    model.eval()
    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            pred = model(images)
            testloss = loss_fn(pred, labels)
            running_test_loss += testloss.item()
            test_n_iter += 1


    avg_train_loss = running_train_loss/train_n_iter
    train_loss.append(avg_train_loss)
    avg_test_loss = running_test_loss/test_n_iter
    test_loss.append(avg_test_loss)
    print("Epoch {}, Train Loss {}, Val Loss {}".format(epoch, avg_train_loss, avg_test_loss))



plt.plot(list(range(EPOCHS)), np.array(train_loss), label="Training L1Error")
plt.plot(list(range(EPOCHS)), np.array(test_loss), label="Test L1Error")
plt.legend()
plt.title("ConvNet-lr{}".format(LEARNING_RATE))
plt.show()



