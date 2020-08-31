import torch
import numpy as np
from tqdm import tqdm
import logging

def step(split, epoch, dataLoader, model, criterion, optimizer, device, numOut, prob='clf'):
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
    tolarance = 0.4 / (numOut - 1)
    for (data, labels) in tqdm(dataLoader):
        # data = data.double()
        if prob == 'reg':
            labels = torch.unsqueeze((labels + 1) * 0.2, 1)
            if device == 'cpu':
                data, labels = data.float(), labels.float()
            else:
                data, labels = data.to(device, dtype=torch.float), labels.to(
                    device, dtype=torch.float)
        else: # clf
            if device == 'cpu':
                data, labels = data.float(), labels.long()
            else:
                data, labels = data.to(device, dtype=torch.float), labels.to(device, dtype=torch.long)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, labels)
        run_loss += loss.item()
        if split == 'train':
            loss.backward()
            optimizer.step()
        else:
            if prob == 'clf':
                _, pred = output.max(1)
                num_true += pred.eq(labels).sum().item()
                total += labels.size(0)
            else: # reg
                num_true += (abs(output - labels) <= tolarance).sum().item()
                print(labels)
                print(output)
                print(num_true)
                total += labels.size(0)
                print(total)

    run_loss /= nIter
    if split == 'val':
        acc = num_true/total
    return loss, acc


def train(epoch, dataLoader, model, criterion, optimizer, device, numOut, prob='clf'):
    return step('train', epoch, dataLoader, model, criterion, optimizer, device, numOut, prob)


def val(epoch, dataLoader, model, criterion, optimizer, device, numOut, prob='clf'):
    return step('val', epoch, dataLoader, model, criterion, optimizer, device, numOut, prob)
