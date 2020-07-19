import torch
import numpy as np
from tqdm import tqdm
import logging

def step(split, epoch, dataLoader, model, criterion, optimizer, device):
    nIter = len(dataLoader)
    if split == 'train':
        model.train()
    else:
        model.eval()
        # print('val len: ', nIter)
    num_true = 0
    total = 0
    acc = 0
    run_loss = 0
    for (data, labels) in tqdm(dataLoader):
        # data = data.double()
        if device == 'cpu':
            data, labels = data.float(), labels.long()
        else:
            data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, labels)
        run_loss += loss.item()
        if split == 'train':
            loss.backward()
            optimizer.step()
        else:
            _, pred = output.max(1)
            num_true += pred.eq(labels).sum().item()
            total += labels.size(0)
    run_loss /= nIter
    if split == 'val':
        acc = num_true/total
    return loss, acc

def train(epoch, dataLoader, model, criterion, optimizer, device):
    return step('train', epoch, dataLoader, model, criterion, optimizer, device)

def val(epoch, dataLoader, model, criterion, optimizer, device):
    return step('val', epoch, dataLoader, model, criterion, optimizer, device)