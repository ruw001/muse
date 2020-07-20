import torch
import torch.utils.data as tud
from model import EEG_CNN
from dataset import EEGDataset
from train import train, val
from resnet import ResNet, BasicBlock, Bottleneck
import logging
import os
import argparse

parser = argparse.ArgumentParser(description='Input for EEG model training')
# args that always change
parser.add_argument('-datasetPath', default='', help='path of your dataset')
parser.add_argument('-exp', default='', help='experiment ID')
parser.add_argument('-useGPU', action='store_true', help='if use GPU or not')
parser.add_argument('-gpuid', type=int, default=0, help='GPU ID')
# args that always change (for testing)
parser.add_argument('-isTest', action='store_true', help='if this is testing or not')
parser.add_argument('-modelPath', default='', help='path for loading the model') #exps/0719_9/model_epoch100.pth

parser.add_argument('-winsize', type=int, default=5, help='window size (s)')
parser.add_argument('-freq', type=int, default=256, help='frequency of signal (Hz)')
parser.add_argument('-stride', type=float, default=0.1, help='stride for generating dataset (s)')
parser.add_argument('-outclass', type=int, default=5, help='number of output classes')

parser.add_argument('-lr', type=float, default=0.002, help='learnign rate')
parser.add_argument('-numEpoch', type=int, default=100, help='# epochs')
parser.add_argument('-batchsize', type=int, default=32, help='batch size')
parser.add_argument('-valInterval', type=int, default=5, help='interval for validation')
parser.add_argument('-signalType', default='EEG', help='specify the type of your signal data')
parser.add_argument('-cnn', default='resnet', help='specify the model you use for training/testing') # vs. 'normal'

args = parser.parse_args()

if not os.path.exists('exps'):
    os.mkdir('exps')

if not os.path.exists(os.path.join('exps', args.exp)):
    os.mkdir(os.path.join('exps', args.exp))

logging.basicConfig(filename=os.path.join('exps', args.exp, 'train.log'), level=logging.INFO)


def saveModel(epoch, model, optimizer, save_path):
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, save_path)


def main(args.isTest):
    logging.info('\nexp name: {}\ndata: {}\ntype: {}\nwinsize={}\nstride={}\nmodel: {}\nlr={}\nbatchsize={}\ngpu={}, {}'\
        .format(args.exp, args.datasetPath, args.signalType, args.winsize, 
        args.stride, args.cnn, args.lr, args.batchsize, args.useGPU, args.gpuid))
    if not args.isTest:
        logging.info('Training start!')
    else:
        logging.info('Testing start!')

    if args.cnn == 'resnet':
        model = ResNet(BasicBlock, [2,2,2,2], num_classes=args.outclass)
    else:
        model = EEG_CNN(args.winsize*args.freq, args.outclass)

    if args.useGPU:
        device = 'cuda:{}'.format(args.gpuid)
        model = model.to(device, dtype=torch.float)
        model = torch.nn.DataParallel(model)
    else:
        device = 'cpu'
        model = model.float()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)

    if not args.isTest:
        dataset = EEGDataset(args.datasetPath, args.signalType, args.freq, args.winsize, args.stride, 'train')
        train_len = int(len(dataset)*0.7)
        trainset, valset = tud.random_split(dataset, [train_len, len(dataset) - train_len])

        train_loader = tud.DataLoader(trainset, batch_size=args.batchsize)
        val_loader = tud.DataLoader(valset, batch_size=args.batchsize)

        for epoch in range(args.numEpoch):
            logging.info('Epoch {} start...'.format(epoch+1))
            print('Epoch {} start...'.format(epoch+1))
            loss, _ = train(epoch, train_loader, model, criterion, optimizer, device)
            logging.info('train loss: {}'.format(loss))
            print('train loss: {}'.format(loss))

            if (epoch+1) % args.valInterval == 0:
                loss, acc = val(epoch, val_loader, model, criterion, optimizer, device)
                print('validation acc: {}, loss: {}'.format(acc, loss))
                logging.info('validation acc: {}, loss: {}'.format(acc, loss))
                saveModel(epoch+1, model, optimizer, os.path.join('exps', args.exp, 'model_epoch{}.pth'.format(epoch+1)))
                logging.info('model saved.')
    else:
        dataset = EEGDataset(args.datasetPath, args.signalType, args.freq, args.winsize, args.stride, 'test')
        test_loader = tud.DataLoader(dataset, batch_size=args.batchsize)
        checkpoint = torch.load(args.modelPath)
        model.load_state_dict(checkpoint['state_dict'])
        logging.info('Model {} loaded!'.format(args.modelPath))
        print('Model {} loaded!'.format(args.modelPath))
        logging.info('Testing start...')
        print('Testing start...')
        loss, acc = val(0, test_loader, model, criterion, optimizer, device)
        print('test acc: {}, loss: {}'.format(acc, loss))
        logging.info('test acc: {}, loss: {}'.format(acc, loss))


main(args.isTest)
