import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nn import *
import matplotlib.pyplot as plt
from torch.autograd import Variable
import scipy.io
import skimage.measure

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

max_iters = 10
batch_size = 16
learning_rate = 1e-3
hidden_size = 64
batches = get_random_batches(train_x, train_y, batch_size)
batch_num = len(batches)

train_x = torch.tensor(train_x).float()
label = np.where(train_y == 1)[1]
label = torch.tensor(label)
train_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(train_x, label),
                                           batch_size=batch_size,
                                           shuffle=True)
valid_x = torch.tensor(valid_x).float()
valid_label = np.where(valid_y == 1)[1]
valid_label = torch.tensor(valid_label)
test_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(valid_x, valid_label),
                                          batch_size=batch_size,
                                          shuffle=True)

train_examples = train_x.shape[0]
valid_examples = valid_x.shape[0]

# model = nn.Sequential(
#     nn.Linear(dimension, hidden_size),
#     nn.Sigmoid(),
#     nn.Linear(hidden_size, classes),
# )


class ConvNet(nn.Module):
    def __init__(self, num_classes=36):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 20, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(20, 50, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Linear(50*5*5, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(-1, 5*5*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


model = ConvNet()

train_loss_data = []
valid_loss_data = []
train_acc_data = []
valid_acc_data = []

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

for i in range(max_iters):
    train_loss = 0
    acc = 0
    valid_loss = 0
    v_acc = 0

    for train_idx, (x, label) in enumerate(train_loader):
        x = x.reshape(batch_size, 1, 32, 32)
        res = model(x)

        loss = criterion(res, label)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        _, pred = torch.max(res.data, 1)
        acc += ((label == pred).sum().item())
        train_loss += loss
    train_acc = acc / train_examples

    for valid_idx, (x, label) in enumerate(test_loader):
        x = x.reshape(batch_size, 1, 32, 32)
        res = model(x)

        loss = criterion(res, label)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        _, pred = torch.max(res.data, 1)
        v_acc += ((label == pred).sum().item())
        valid_loss += loss
    valid_acc = v_acc / valid_examples

    train_loss_data.append(train_loss/(train_examples/batch_size))
    valid_loss_data.append(valid_loss/(valid_examples/batch_size))
    train_acc_data.append(train_acc)
    valid_acc_data.append(valid_acc)

    print('Validation accuracy: ', valid_acc)
    if i % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(
            i, train_loss, train_acc))

plt.figure(0)
plt.plot(np.arange(max_iters), train_acc_data, 'r')
plt.plot(np.arange(max_iters), valid_acc_data, 'b')
plt.legend(['training accuracy', 'valid accuracy'])
plt.show()
plt.figure(1)
plt.plot(np.arange(max_iters), train_loss_data, 'r')
plt.plot(np.arange(max_iters), valid_loss_data, 'b')
plt.legend(['training loss', 'valid loss'])
plt.show()
