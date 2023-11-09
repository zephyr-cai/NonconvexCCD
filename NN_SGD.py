import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import argparse
import sys
import time
# import tracemalloc

from utils import set_random_seed, write_res
from torch import Tensor
from typing import Union, List, Dict, Any, cast, Type, Callable, Optional

# LeNet 
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # For MNIST dataset
        # self.conv1 = spectral_norm(nn.Conv2d(1, 6, 5))
        # self.conv2 = spectral_norm(nn.Conv2d(6, 16, 5))
        # self.fc1   = spectral_norm(nn.Linear(256, 120))
        # self.fc2   = spectral_norm(nn.Linear(120, 84))
        # self.fc3   = spectral_norm(nn.Linear(84, 10))
        # self.conv1 = nn.Conv2d(1, 6, 5)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1   = nn.Linear(256, 120)
        # self.fc2   = nn.Linear(120, 84)
        # self.fc3   = nn.Linear(84, 10)

        # For CIFAR-10 dataset
        # self.conv1 = nn.Conv2d(3, 6, 5)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1   = spectral_norm(nn.Linear(16*5*5, 120))
        # self.fc2   = spectral_norm(nn.Linear(120, 84))
        # self.fc3   = spectral_norm(nn.Linear(84, 10))
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MNIST test for optimization algorithms')
    parser.add_argument('--epoch', '-e', default=100, type=int, help='number of epochs for testing')
    parser.add_argument('--lr', '-l', default=0.1, type=float, help='learning rate')
    parser.add_argument('--final-lr', '-ll', default=0.01, type=float, help='final learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--gamma', '-g', default=0.1, type=float, help='learing rate step gamma')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--schedule', type=int, nargs='+', default=[80, 120, 160], help='Decrease learning rate at these epochs.')
    parser.add_argument('--train-batch', default=256, type=int, help='train batchsize')
    parser.add_argument('--test-batch', default=100, type=int, help='test batchsize')
    parser.add_argument('--seed', '-s', default=0, type=int, help='random seed')
    args = parser.parse_args()

    set_random_seed(args.seed)
    torch.set_num_threads(1)
    
    print(0)
    
    device = 'cuda' if (torch.cuda.is_available()) else 'cpu'

    print("preparing data...")
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    # data_train = datasets.MNIST(
    #     root="./data_cache/",
    #     transform=transforms.ToTensor(),
    #     train=True,
    #     download=True
    # )
    # data_test = datasets.MNIST(
    #     root="./data_cache/",
    #     transform=transforms.ToTensor(),
    #     train=False
    # )
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    data_train = datasets.CIFAR10(
        root="./data_cache/",
        transform=transform_train,
        train=True,
        download=True
    )
    data_test = datasets.CIFAR10(
        root="./data_cache/",
        transform=transform_test,
        train=False
    )

    trainloader = DataLoader(data_train, batch_size=args.train_batch, shuffle=True)
    testloader = DataLoader(data_test, batch_size=args.test_batch, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()

    model = LeNet()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=args.final_lr)

    log_loss = list()
    log_acc = list()
    log_time = list()

    model.to(device)
    criterion = criterion.to(device)
    print("training...")
    time_start = time.time()
    for epoch in range(args.epoch):
        model.train()
        agg_loss = 0
        t1 = time.time()
        for train_batch, (data, label) in enumerate(trainloader):
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            pred = model(data)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            agg_loss += loss.item()
        t2 = time.time()
        log_time.append(t2 - t1)
        
        model.eval()
        acc_sum = 0
        loss_test = 0
        acc_train = 0
        with torch.no_grad():
            for val_batch, (data_val, label_val) in enumerate(testloader):
                data_val, label_val = data_val.to(device), label_val.to(device)
                pred = model(data_val)
                _, predicted = torch.max(pred, 1)
                acc_sum += (predicted == label_val).sum().item()
            agg_loss /= len(trainloader)
            acc_sum /= len(testloader.dataset)
            log_loss.append(agg_loss)
            log_acc.append(acc_sum)
            print("epoch: {:d} | loss: {:f} | acc: {:f}".format(epoch, agg_loss, acc_sum))

    time_end = time.time()
    print("epoch time", time_end - time_start)
    log_time = np.array(log_time)
    print("mean time: ", log_time.mean(), "std time: ", log_time.std())
    write_res('./results/log_loss_SGD_cifar_lenet_' + str(args.lr) + '_' + str(args.train_batch) + '_new3.npy', log_loss)
    write_res('./results/log_acc_SGD_cifar_lenet_' + str(args.lr) + '_' + str(args.train_batch) + '_new3.npy', log_acc)
    print("Results written!")