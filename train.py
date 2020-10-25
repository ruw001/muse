import torch
import numpy as np
from tqdm import tqdm
import logging

def step(split, epoch, dataLoader, model, criterion, optimizer, device, numOut, prob='clf', printout=False):
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
    tolarance = 0.5 / (numOut - 1)
    gt_list = []
    pred_list = []
    for (data, labels) in tqdm(dataLoader):
        # data = data.double()
        if prob == 'reg':
            labels = torch.unsqueeze(labels * (1/(numOut -1)), 1)
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
                if printout:
                    gt_list.append(labels)
                    pred_list.append(pred)
            else: # reg
                num_true += (abs(output - labels) < tolarance).sum().item()
                # print(labels)
                # print(output)
                # print(num_true)
                total += labels.size(0)
                # print(total)

    if printout:
        with open('out.txt', 'w') as outfile:
            gt_list = torch.cat(gt_list)
            pred_list = torch.cat(pred_list)
            for i in range(gt_list.size(0)):
                _label = gt_list[i].item()
                _pred = pred_list[i].item()
                outfile.write(str(_label) + ',' + str(_pred) + '\n')

    run_loss /= nIter
    if split == 'val':
        acc = num_true/total
    return loss, acc


def train(epoch, dataLoader, model, criterion, optimizer, device, numOut, prob='clf'):
    return step('train', epoch, dataLoader, model, criterion, optimizer, device, numOut, prob)


def val(epoch, dataLoader, model, criterion, optimizer, device, numOut, prob='clf', printout=False):
    return step('val', epoch, dataLoader, model, criterion, optimizer, device, numOut, prob, printout=printout)
