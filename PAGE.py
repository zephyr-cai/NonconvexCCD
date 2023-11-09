import math
import re
import numpy as np
import torch
import random
import copy
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse
import sys
import time
import copy

from utils import var, softmax, launch_a_process, synchronize, setup, set_random_seed, write_res, PageLoader
from NN_SGD import LeNet

def main(rank, args):
    print(0)
    set_random_seed(args['seed'])
    torch.set_num_threads(1)

    device = 'cuda' if (torch.cuda.is_available()) else 'cpu'

    base_lr = args['lr']
    wd = args['weight_decay']
    T_max = args['epoch']
    final_lr = args['final_lr']

    if rank == 0:
        print("Preparing data...")
    if args['dataset'] == "mnist":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
        data_train = datasets.MNIST(
            root="./data_cache/",
            transform=transform,
            train=True,
            download=True
        )
        data_test = datasets.MNIST(
            root="./data_cache/",
            transform=transform,
            train=False
        )
    elif args['dataset'] == 'cifar':
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

    if args['model'] == "vgg":
        model = models.vgg16_bn(pretrained=False)
        model_prev = models.vgg16_bn(pretrained=False)
    elif args['model'] == "resnet":
        model = models.resnet18(pretrained=False)
        model_prev = models.resnet18(pretrained=False)
    elif args['model'] == "lenet":
        model = LeNet()
        model_prev = LeNet()
    g_est = dict()
    with torch.no_grad():
        for idx, (param, param_prev) in enumerate(zip(model.parameters(), model_prev.parameters())):
            param_prev.data.copy_(param.data)
            g_est[idx] = torch.zeros_like(param.data).to(device)

    log_loss_train = list()
    log_acc_test = list()
    log_time = list()
    
    batch_large = args['train_batch']
    batch_small = int(np.sqrt(batch_large))
    p = batch_small / (batch_large + batch_small)
    print("batch: ", batch_small, batch_large)

    trainloader = DataLoader(data_train, batch_size=batch_small, shuffle=True, drop_last=True)
    # trainloader = PageLoader(data_train, large_batch=batch_large, small_batch=batch_small, p=p)
    testloader = DataLoader(data_test, batch_size=args['test_batch'], shuffle=False, drop_last=True)
    criterion = torch.nn.CrossEntropyLoss()
    
    # optimizing...
    if rank == 0:
        print("Training...")
    flag_batch = True
    flag_toss = True
    cnt = 0
    model.to(device)
    model_prev.to(device)
    criterion = criterion.to(device)
    train_cnt_total = 0
    time_start = time.time()
    t = 0
        
    for epoch in range(args['epoch']):
        lr = final_lr + (base_lr - final_lr) * (1 + math.cos(math.pi * epoch / T_max)) / 2
        # lr = base_lr
        agg_loss = 0
        train_cnt = 0
        model.train()
        model_prev.train()
        t1 = time.time()
        for train_batch, (data, label) in enumerate(trainloader):
            if flag_toss:
                toss = np.random.binomial(1, p)
                flag_batch = True if toss == 1 else False
                flag_toss = False
            if train_cnt == 0 and epoch == 0:
                flag_batch = True
            if flag_batch:
                if cnt == 0:
                    batch_data, batch_label = data, label
                    cnt += 1
                else:
                    batch_data = torch.cat((batch_data, data), 0)
                    batch_label = torch.cat((batch_label, label), 0)
                    cnt += 1
                    if cnt == batch_small:
                        batch_data, batch_label = batch_data.to(device), batch_label.to(device)
                        model.zero_grad()
                        prediction = model(batch_data)
                        loss = criterion(prediction, batch_label)
                        loss.backward()
                        agg_loss += loss.item()
                        with torch.no_grad():
                            for idx, (param, param_prev) in enumerate(zip(model.parameters(), model_prev.parameters())):
                                if param.grad is None:
                                    continue
                                else:
                                    param_prev.data.copy_(param.data)
                                    dp = param.grad.data + wd * param.data
                                    g_est[idx] = torch.clone(dp)
                                    param.data.add_(dp, alpha=-lr)
                        train_cnt += 1
                        flag_batch = False
                        cnt = 0
                        flag_toss = True
            else:
                data, label = data.to(device), label.to(device)
                model.zero_grad()
                prediction = model(data)
                loss = criterion(prediction, label)
                loss.backward()
                model_prev.zero_grad()
                prediction_prev = model_prev(data)
                loss_prev = criterion(prediction_prev, label)
                loss_prev.backward()
                with torch.no_grad():
                    for idx, (param, param_prev) in enumerate(zip(model.parameters(), model_prev.parameters())):
                        if param.grad is None:
                            continue
                        else:
                            dp, dpp = param.grad.data + wd * param.data, param_prev.grad.data + wd * param_prev.data
                            param_prev.data.copy_(param.data)
                            dp = dp.add(dpp, alpha=-1)
                            dp = dp.add(g_est[idx], alpha=1)
                            g_est[idx] = torch.clone(dp)
                            param.data.add_(dp, alpha=-lr)
                agg_loss += loss.item()
                train_cnt += 1
                flag_toss = True
                flag_batch = False
        t2 = time.time()
        log_time.append(t2 - t1)
        # test
        acc_sum = 0
        model.eval()
        for val_batch, (data_val, label_val) in enumerate(testloader):
            data_val, label_val = data_val.to(device), label_val.to(device)
            pred = model(data_val)
            _, predicted = torch.max(pred, 1)
            acc_sum += (predicted == label_val).sum().item()
        agg_loss /= train_cnt
        acc_sum /= len(testloader.dataset)
        log_acc_test.append(acc_sum)
        log_loss_train.append(agg_loss)
        train_cnt_total += train_cnt
        if rank == 0:
            print("epoch: {:d} | loss: {:f} | validation precision: {:f}".format(epoch, agg_loss, acc_sum))

    if rank == 0:
        time_end = time.time()
        print("epoch time", time_end - time_start)
        log_time = np.array(log_time)
        print("mean time: ", log_time.mean(), "std time: ", log_time.std(), "iteration time: ", log_time.sum()/train_cnt_total)
        write_res('./results/log_loss_PAGE_' + str(args['lr']) + '_' + str(args['dataset']) + '_' + str(args['train_batch']) + '_new4.npy', log_loss_train)
        write_res('./results/log_acc_PAGE_' + str(args['lr']) + '_' + str(args['dataset']) + '_' + str(args['train_batch']) + '_new4.npy', log_acc_test)
        print("Results written!")
        
    sys.exit("break!")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Numerical Test for Consensus-based Global Optimization Method',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # problem setting
    parser.add_argument('-d', '--dataset', type=str, default=None, help='the dataset we use')
    parser.add_argument('-m', '--model', type=str, default=None, help='the NN model we use')
    parser.add_argument('--epoch', '-e', default=100, type=int, help='number of epochs for testing')
    parser.add_argument('--lr', '-l', default=0.1, type=float, help='learning rate')
    parser.add_argument('--final-lr', '-ll', default=0.01, type=float, help='final learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--gamma', '-g', default=0.1, type=float, help='learing rate step gamma')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--schedule', type=int, nargs='+', default=[80, 120, 160], help='Decrease learning rate at these epochs.')
    parser.add_argument('--train-batch', default=256, type=int, help='train batchsize')
    parser.add_argument('--test-batch', default=100, type=int, help='test batchsize')
    
    # general setting
    parser.add_argument('-s', '--seed', type=int, default=0, help='the random seed')
    parser.add_argument('-gp', '--gpu', action="store_true", help='whether to apply gpu training')
    parser.add_argument('-npp', '--num-processes', type=int, default=1, help='the number of processes for multiprocessing optimization')
    parser.add_argument('-mi', '--master-ip', type=str, default='127.0.0.1')
    parser.add_argument('-mp', '--master-port', type=str, default='12345')
    
    args = parser.parse_args()
    args = setup(args)

    if args['num_processes'] == 1:
        main(0, args)
    else:
        mp = torch.multiprocessing.get_context('spawn')
        procs = []
        for rank in range(args['num_processes']):
            procs.append(mp.Process(target=launch_a_process, args=(rank, args, main), daemon=True))
            procs[-1].start()
        for p in procs:
            p.join()

        

