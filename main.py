import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import datetime
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-config", "--config", help="run config file", type=str)

import json
import torchvision

from torchvision.transforms import transforms

from utils import CIFAR10Noisy, CIFAR100Noisy
from torchvision.datasets.cifar import CIFAR10, CIFAR100

from models import resnet18k

'''
parser = argparse.ArgumentParser(description='Double Descent Training')
parser.add_argument('--epochs', default=4000, type=int, metavar='N',
                    help='Number of Epochs to Run For')
parser.add_argument('--optimizer', default='Adam', type=str, metavar='OPTIM')
parser.add_argument('--pretrain
ed', default=False, type=bool, metavar='BOOL')
parser.add_argument('--gpu_num', default=0, type=int, metavar='N')
parser.add_argument('--dataset', default='cifar10', type=str, metavar='DATASET')
parser.add_argument('--lr', default=0.0001, type=float, metavar='LR')

parser.add_argument('--config', default='/configs/resnet-18_epoch_dd.json', type='str', metavar='PATH')
'''


def noise_val(noise_str):
    i = int(noise_str)
    return float(i / 100)


def num_classes(dataset):
    if dataset == 'cifar10':
        return 10
    if dataset == 'cifar100':
        return 100


def load_dataset(dataset, train, transforms=None, noise=None):
    if dataset == 'cifar10':
        if noise is None or not train:
            return CIFAR10('./data/cifar-10/', train=train, transform=transforms)
        else:
            return CIFAR10Noisy('./data/cifar-10-' + noise, train=train, transform=transforms)
    elif dataset == 'cifar100':
        if noise is None or not train:
            return CIFAR100('./data/cifar-100/', train=train, transform=transforms)
        else:
            return CIFAR100Noisy('./data/cifar-100-' + noise, train=train, transform=transforms)
    raise ValueError(dataset + ' is not a valid dataset.')


def init_model(model, k, classes):
    # TODO: add support for CNN, resnet-50, resnet-101, and wide resnet
    if model == 'resnet-18':
        return resnet18k.make_resnet18k(k=k, num_classes=classes)
    else:
        raise ValueError(model + ' is not a valid model.')


def init_optimizer(model, optim, lr):
    if optim == 'adam':
        return optim.Adam(model.parameters(), lr=lr)
    else:
        # this involves a weird scheduler
        raise NotImplementedError('SGD not implemented')
        #return optim.SGD(model.parameters(), lr=lr)


def init_scheduler(optim):
    if optim == 'sgd':
        # TODO: Add a proper learning rate scheduler here.
        return None
    return None


def get_norm_params(dataset):
    if dataset == 'cifar10':
        return (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)


def gen_timestamp():
    now = datetime.datetime.now()
    year = '{:02d}'.format(now.year)
    month = '{:02d}'.format(now.month)
    day = '{:02d}'.format(now.day)
    hour = '{:02d}'.format(now.hour)
    minute = '{:02d}'.format(now.minute)
    sec = '{:02d}'.format(now.second)
    return '{}-{}-{}-{}-{}-{}'.format(year, month, day, hour, minute, sec)


def gen_uid(config):
    f = 'res_' + config['dataset'] + '_' + config['model'] + '_' \
        + str(config['k']) + '_' \
        + config['dataset'] + '_'

    if 'noise' in config:
        f += str(config['noise']) + '_'

    if 'data_aug' in config:
        f += str(config['data_aug']) + '_'

    f += gen_timestamp() + '.json'
    return f


def train(model, trainloader, optim, criterion, epochs, res, n_samples):
    running_loss = 0.0
    epoch_loss = 0.0
    res['train_epoch_stats'] = {}
    for epoch in epochs:
        print('\n\n---')
        print('Starting epoch: ' + epoch)

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            else:
                inputs = inputs.cpu()
                labels = labels.cpu()

            optim.zero_grad()

            pred = model(inputs)
            loss = criterion(pred, labels)
            loss.backward()
            optim.step()

            epoch_loss += loss.item()
            running_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
        print(epoch + 1, epoch_loss / len(n_samples))
        res['epoch_stats'][epoch]['loss'] = epoch_loss / len(n_samples)
        res['epoch_stats'][epoch]['acc'] = epoch_loss / len(n_samples)
        print('---')


def test(model, testloader, criterion, res, n_samples):
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            test_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    res['test_stats']['loss'] = test_loss / n_samples
    res['test_stats']['acc'] = 100 * correct / total

    print('Test loss: ' + str(test_loss))
    print('Accuracy: %d %%' % (
            100 * correct / total))


def run(config):
    run_uid = gen_uid(config)
    res = {}
    res['config'] = config

    model = init_model(config['model'], k=config['k'], classes=num_classes(config['dataset']))

    lr = config['lr'] if 'lr' in config else 0.0001
    optim = config['optim'] if 'optim' in config else 'adam'
    optim = init_optimizer(model, optim, lr=lr)

    # start with a small number of epochs
    epochs = config['epochs'] if 'epochs' in config else 300
    noise = config['noise'] if 'noise' in config else None

    # this shouldn't be changed
    batch_size = config['batch_size'] if 'batch_size' in config else 128

    # use data augmentation by default
    data_aug = config['data_aug'] if 'data_aug' in config else True

    if data_aug:
        train_transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(*get_norm_params(config['dataset']))])
    else:
        train_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(*get_norm_params(config['dataset']))])

    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(*get_norm_params(config['dataset']))])

    train_dataset = load_dataset(dataset=config['dataset'],
                                 train=True,
                                 noise=config['noise'] if 'noise' in config else None,
                                 transforms=train_transform)

    test_dataset = load_dataset(dataset=config['dataset'],
                                train=False,
                                noise=config['noise'] if 'noise' in config else None,
                                transforms=test_transform)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                             shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model.cpu()

    print('\n\n---TRAINING---\n')
    train(model, trainloader, optim, epochs, res, len(train_dataset))

    # TODO: this will probably fail on the grid
    torch.save(model.state_dict(), 'trained/' + run_uid + '.pth')

    print('\n\n---TESTING---\n')
    test(model, testloader, criterion, res, len(test_dataset))

    with open(run_uid, 'rw') as f:
        json.dump(res, f)


if __name__ == '__main__':
    args = sys.argv[1:]
    options = parser.parse_args(args)

    with open(options.config, 'r') as f:
        config = json.load(f)
        run(config)
