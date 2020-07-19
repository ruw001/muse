import torch
import torch.utils.data as tud
from model import EEG_CNN
from dataset import EEGDataset
from train import train, val
from resnet import ResNet, BasicBlock, Bottleneck
import logging
import os

dataset_path = 'data/ru0718train'
winsize = 5
freq = 256
stride = 0.1
outclass = 5
lr = 0.002
numEpoch = 200
batchsize = 2
val_interval = 2
type_ = 'EEG'
cnn = 'resnet' # vs. 'normal'
exp = '0718_x'
use_gpu = False
gpuid = 0

if not os.path.exists(os.path.join('exps', exp)):
    os.mkdir(os.path.join('exps', exp))

logging.basicConfig(filename=os.path.join('exps', exp, 'train.log'), level=logging.INFO)


def saveModel(epoch, model, optimizer, save_path):
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, save_path)


def main():
    logging.info('\nexp name: {}\ndata: {}\ntype: {}\nwinsize={}\nstride={}\nmodel: {}\nlr={}\nbatchsize={}\ngpu={}, {}'\
        .format(exp, dataset_path, type_, winsize, stride, cnn, lr, batchsize, use_gpu, gpuid))
    logging.info('Training start!')

    if cnn == 'resnet':
        model = ResNet(BasicBlock, [2,2,2,2], num_classes=outclass)
    else:
        model = EEG_CNN(winsize*freq, outclass)

    if use_gpu:
        device = 'cuda:{}'.format(gpuid)
        model = model.to(device)
        model = torch.nn.DataParallel(model)
    else:
        device = 'cpu'
        model = model.float()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

    dataset = EEGDataset(dataset_path, type_, freq, winsize, stride, 'train')
    train_len = int(len(dataset)*0.7)
    trainset, valset = tud.random_split(dataset, [train_len, len(dataset) - train_len])

    train_loader = tud.DataLoader(trainset, batch_size=batchsize)
    val_loader = tud.DataLoader(valset, batch_size=batchsize)

    for epoch in range(numEpoch):
        logging.info('Epoch {} start...'.format(epoch+1))
        print('Epoch {} start...'.format(epoch+1))
        loss, _ = train(epoch, train_loader, model, criterion, optimizer, device)
        logging.info('train loss: {}'.format(loss))
        print('train loss: {}'.format(loss))

        if (epoch+1) % val_interval == 0:
            loss, acc = val(epoch, val_loader, model, criterion, optimizer, device)
            print('validation acc: {}, loss: {}'.format(acc, loss))
            logging.info('validation acc: {}, loss: {}'.format(acc, loss))
            saveModel(epoch+1, model, optimizer, os.path.join('exps', exp, 'model_epoch{}.pth'.format(epoch+1)))
            logging.info('model saved.')

main()
