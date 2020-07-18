import torch
import numpy as np
from progress.bar import Bar


def step(split, epoch, dataLoader, model, criterion, optimizer):
    model = model.float()
    if split == 'train':
        model.train()
    else:
        model.eval()
    nIter = len(dataLoader)
    bar = Bar("exp", max=nIter)
    for i, (data, labels) in enumerate(dataLoader):
        # data = data.double()
        output = model(data.float())
        loss = criterion(output, labels.long())
        acc = 0
        if split == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            _, pred = torch.max(output, 1)
            acc = np.sum(pred.numpy() == labels.numpy())/labels.shape[0]
        Bar.suffix = '{} Epoch: [{}][{}/{}]| Total: {} | ETA: {} | Loss {} | Acc {}'\
            .format(split, epoch, i, nIter, bar.elapsed_td, bar.eta_td, loss, acc)
        return loss, acc

def train(epoch, dataLoader, model, criterion, optimizer):
    return step('train', epoch, dataLoader, model, criterion, optimizer)

def val(epoch, dataLoader, model, criterion, optimizer):
    return step('val', epoch, dataLoader, model, criterion, optimizer)