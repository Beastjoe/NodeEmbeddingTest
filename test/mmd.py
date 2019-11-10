import torch
from torch.autograd import Variable
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''

    :param source: samples from source. Size: [batch size, sample size, vector size]
    :param target: samples from target. Size: [batch size, sample size, vector size]
    :param kernel_mul:
    :param kernel_num:
    :param fix_sigma:
    :return:
    '''
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)), int(total.size(2)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)), int(total.size(2)))
    L2_distance = ((total0 - total1) ** 2).sum([2, 3])
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  # /len(kernel_val)


def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss

def standardize(data):
    dim = len(data[0])
    min_max = []
    for i in range(dim):
        min_max.append([1e10, -1e10])
    for node in data:
        for i in range(dim):
            min_max[i][0] = min(node[i], min_max[i][0])
            min_max[i][1] = max(node[i], min_max[i][1])
    for node in data:
        for i in range(dim):
            node[i] = (node[i] - min_max[i][0]) / (min_max[i][1] - min_max[i][0])

def mmd_test(source, target, batch_size, sample_size):
    standardize(source)
    standardize(target)

    plt.subplot(1, 2, 1)
    plt.xlabel("source")
    X = []
    Y = []
    for node in source:
        X.append(node[0])
        Y.append(node[1])
    plt.scatter(X, Y)

    plt.subplot(1, 2, 2)
    plt.xlabel("target")
    X = []
    Y = []
    for node in target:
        X.append(node[0])
        Y.append(node[1])
    plt.scatter(X, Y)
    plt.show()

    source_sample = []
    for i in range(batch_size):
        source_sample.append(random.sample(source, k=sample_size))
    target_sample = []
    for i in range(batch_size):
        target_sample.append(random.sample(target, k=sample_size))
    X = torch.Tensor(source_sample)
    Y = torch.Tensor(target_sample)
    X, Y = Variable(X), Variable(Y)
    print("MMD score:" + str(mmd_rbf(X, Y)))
