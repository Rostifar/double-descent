import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import models.resnet18k as resnet

from torch.utils.tensorboard import SummaryWriter


#k = [1, 2, 4, 8, 10, 20, 30, 50, 60, 64]

# TODO: determine whether this normalization is fine

train_transform = transforms.Compose(
    [transforms.RandomCrop(32, padding=4),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2)


test_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)


model = resnet.make_resnet18k(k=64, num_classes=10)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()


if torch.cuda.is_available():
    model = model.cuda()
else:
    model = model.cpu()

for epoch in range(10):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        else:
            inputs = inputs.cpu()
            labels = labels.cpu()

        optimizer.zero_grad()

        pred = model(inputs)
        loss = criterion(pred, labels)
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    print('Finished Training')


PATH = './cifar_net.pth'
torch.save(model.state_dict(), PATH)


correct = 0
total = 0
test_err = 0.0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        test_err += criterion(outputs, labels).item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    test_err /= len(testloader)
print('Test err: ' + str(test_err))
print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
