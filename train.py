import torch
import numpy as np
from tqdm import tqdm
import logging

def step(split, epoch, dataLoader, model, criterion, optimizer):
    model = model.float()
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
        output = model(data.float())
        loss = criterion(output, labels.long())
        run_loss += loss.item()
        if split == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            _, pred = torch.max(output, 1)
            num_true += np.sum(pred.numpy() == labels.numpy())
            total += labels.shape[0]
    run_loss /= nIter
    if split == 'val':
        acc = num_true/total
    return loss, acc

def train(epoch, dataLoader, model, criterion, optimizer):
    return step('train', epoch, dataLoader, model, criterion, optimizer)

def val(epoch, dataLoader, model, criterion, optimizer):
    return step('val', epoch, dataLoader, model, criterion, optimizer)