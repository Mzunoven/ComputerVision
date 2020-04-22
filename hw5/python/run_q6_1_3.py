# -*- coding: utf-8 -*-

# import package #
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

""" 
Loading dataset:
In this case, apply flip and grayscale randomly to training dataset in
order to prevent overfitting and improve accuracy
"""
transform1 = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.RandomGrayscale(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform2 = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform1)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=20, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform2)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

"""
Define Convolution Neural Network and criterion
"""


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.MaxPool2d(2, 2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.MaxPool2d(2, 2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.MaxPool2d(2, 2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.MaxPool2d(2, 2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.fc1 = nn.Linear(512*2*2, 120)
        self.drop1 = nn.Dropout2d()
        self.fc2 = nn.Linear(120, 84)
        self.drop2 = nn.Dropout2d()
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = self.fc3(x)
        return x


net = CNN()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
device = torch.device("cuda:0"if torch.cuda.is_available() else "cpu")

"""
Training Process:
epochs = 6
batch_size = 20
"""

batch_size = 20
train_loss_data = []
train_acc_data = []

num_epochs = 6
total_step = len(trainloader)
for e in range(num_epochs):
    train_loss = 0
    acc = 0
    total = 0

    for i, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)
        output = net(images)
        loss = criterion(output, labels)
        total += labels.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, pred = torch.max(output.data, 1)
        acc += (pred == labels).sum().item()
        train_loss += loss

        if (i+1) % 2500 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(e +
                                                                      1, num_epochs, i+1, total_step, loss.item()))
    train_loss_data.append(train_loss/(total/batch_size))
    train_acc_data.append(acc/total)

"""
Test Process: final test accuracy is 83.46%
"""
net.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        output = net(images)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy: {} %'.format(100 * correct / total))

plt.figure(0)
plt.plot(np.arange(num_epochs), train_acc_data, 'r')
plt.legend(['training accuracy'])
plt.show()
plt.figure(1)
plt.plot(np.arange(num_epochs), train_loss_data, 'r')
plt.legend(['training loss'])
plt.show()
