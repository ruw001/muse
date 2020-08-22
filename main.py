import torch
import torch.utils.data as tud
from model import EEG_CNN
from dataset import EEGDataset
from train import train, val
from resnet import ResNet, BasicBlock, Bottleneck
from earlystop import EarlyStopping
import logging
import os
import argparse

parser = argparse.ArgumentParser(description='Input for EEG model training')
# args that always change
parser.add_argument('-datasetPath', default='', help='path of your dataset')
parser.add_argument('-exp', default='', help='experiment ID')
parser.add_argument('-useGPU', action='store_true', help='if use GPU or not')
parser.add_argument('-gpuids', nargs='+', type=int,
                    default=[0], help='GPU IDs')
parser.add_argument('-outclass', nargs='+', type=int,
                    default=[0, 1, 2, 3, 4], help='output classes')
parser.add_argument('-cv', action='store_true',
                    help='if this exp uses cross validation')

# args that always change (for testing)
parser.add_argument('-isTest', action='store_true',
                    help='if this is testing or not')
# exps/0719_9/model_epoch100.pth
parser.add_argument('-modelPath', default='',
                    help='path for loading the model')

parser.add_argument('-winsize', type=int, default=5, help='window size (s)')
parser.add_argument('-freq', type=int, default=256,
                    help='frequency of signal (Hz)')
parser.add_argument('-stride', type=float, default=0.1,
                    help='stride for generating dataset (s)')

parser.add_argument('-E', action='store_true', help='extract freq bands and use FFT')

parser.add_argument('-lr', type=float, default=0.002, help='learnign rate')
parser.add_argument('-numEpoch', type=int, default=200, help='# epochs')
parser.add_argument('-batchsize', type=int, default=32, help='batch size')
parser.add_argument('-valInterval', type=int, default=1,
                    help='interval for validation')
parser.add_argument('-signalType', default='EEG',
                    help='specify the type of your signal data')
parser.add_argument('-cnn', default='resnet',
                    help='specify the model you use for training/testing')  # vs. 'normal'
parser.add_argument('-patience', type=int, default=5, help='patience for early stopper')

opt = parser.parse_args()

if not os.path.exists('exps'):
    os.mkdir('exps')

if not os.path.exists(os.path.join('exps', opt.exp)):
    os.mkdir(os.path.join('exps', opt.exp))

logging.basicConfig(filename=os.path.join(
    'exps', opt.exp, 'train.log'), level=logging.INFO)


def saveModel(epoch, model, optimizer, save_path):
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, save_path)


def main(isTest):
    logging.info('\nexp name: {}\ndata: {}\ntype: {}\noutclass={}\nwinsize={}\nstride={}\nmodel: {}\nlr={}\nbatchsize={}\ngpu={}, {}'
                 .format(opt.exp, opt.datasetPath, opt.signalType, opt.outclass, opt.winsize,
                         opt.stride, opt.cnn, opt.lr, opt.batchsize, opt.useGPU, opt.gpuids))
    if not opt.isTest:
        logging.info('Training start!')
    else:
        logging.info('Testing start!')

    if opt.cnn == 'resnet':
        model = ResNet(8 if opt.E else 4, BasicBlock, [2, 2, 2, 2], num_classes=len(opt.outclass))
    else:
        assert opt.E == False
        model = EEG_CNN(opt.winsize*opt.freq, len(opt.outclass))

    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=opt.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    initEpoch = 0

    if opt.useGPU:
        device = 'cuda:{}'.format(opt.gpuids[0])
        model = model.to(device, dtype=torch.float)
        model = torch.nn.DataParallel(model, device_ids=opt.gpuids)
        criterion = criterion.to(device)
    else:
        device = 'cpu'
        model = model.float()

    if opt.modelPath:
        logging.info(
            'Loading model parameters from {}...'.format(opt.modelPath))
        print('Loading model parameters from {}...'.format(opt.modelPath))
        checkpoint = torch.load(opt.modelPath)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        initEpoch = checkpoint['epoch']

    if not opt.isTest and not opt.cv:
        dataset = EEGDataset(os.path.join(opt.datasetPath, 'train'),
                             opt.signalType, opt.freq, opt.winsize, opt.stride, 'train', opt.E, opt.outclass)
        train_len = int(len(dataset)*0.7)
        trainset, valset = tud.random_split(
            dataset, [train_len, len(dataset) - train_len])
        testset = EEGDataset(os.path.join(opt.datasetPath, 'test'),
                             opt.signalType, opt.freq, opt.winsize, opt.stride, 'test', opt.E, opt.outclass)

        train_loader = tud.DataLoader(trainset, batch_size=opt.batchsize)
        print('trainset size:', len(train_loader))
        val_loader = tud.DataLoader(valset, batch_size=opt.batchsize)
        print('valset size:', len(val_loader))
        test_loader = tud.DataLoader(testset, batch_size=opt.batchsize)
        print('testset size:', len(test_loader))
        val_acc = 0
        acc = 0
        early_stopper = EarlyStopping(patience=opt.patience, verbose=True) #TODO: make patience a tunable parameter
        for epoch in range(initEpoch, initEpoch + opt.numEpoch):
            logging.info('Epoch {} start...'.format(epoch+1))
            print('Epoch {} start...'.format(epoch+1))
            loss, _ = train(epoch, train_loader, model,
                            criterion, optimizer, device)
            logging.info('train loss: {}'.format(loss))
            print('train loss: {}'.format(loss))

            if (epoch+1) % opt.valInterval == 0:
                loss, acc = val(epoch, val_loader, model,
                                criterion, optimizer, device)
                logging.info('validation acc: {}, loss: {}'.format(acc, loss))
                print('validation acc: {}, loss: {}'.format(acc, loss))
                
                decision = early_stopper(loss)

                logging.info('Testing start...')
                print('Testing start...')
                loss, t_acc = val(epoch, test_loader, model,
                                  criterion, optimizer, device)
                print('test acc: {}, loss: {}'.format(t_acc, loss))
                logging.info('test acc: {}, loss: {}'.format(t_acc, loss))

                if decision == 'save':
                    saveModel(epoch+1, model, optimizer, os.path.join('exps',
                                                                    opt.exp, 'model_epoch{}.pth'.format(epoch+1)))
                    logging.info('model saved.')
                    print('model saved.')
                elif decision == 'stop':
                    logging.info('early stopped!')
                    print('early stopped!')
                    break         
                
    elif opt.cv:
        cvsets = []
        testset = []
        # relative dir!
        folders = [f for f in os.listdir(opt.datasetPath) if os.path.isdir(
            os.path.join(opt.datasetPath, f))]
        testIdx = max(folders)
        for f in folders:
            if f != testIdx:
                dataset = EEGDataset(os.path.join(
                    opt.datasetPath, f), opt.signalType, opt.freq, opt.winsize, opt.stride, 'train', opt.E, opt.outclass)
                cvsets.append(dataset)
            else:
                testset = EEGDataset(os.path.join(
                    opt.datasetPath, f), opt.signalType, opt.freq, opt.winsize, opt.stride, 'test', opt.E, opt.outclass)
        
        early_stopper = EarlyStopping(patience=opt.patience, verbose=True)
        # cvloaders = [tud.DataLoader(d, batch_size=opt.batchsize) for d in cvsets]
        test_loader = tud.DataLoader(testset, batch_size=opt.batchsize)
        for cluster in range(initEpoch, initEpoch + opt.numEpoch, len(cvsets)):
            avg_val_loss = 0
            for i in range(len(cvsets)):
                train_dataset = tud.ConcatDataset(
                    [cvsets[l] for l in range(len(cvsets)) if l != i])
                train_loader = tud.DataLoader(train_dataset, batch_size=opt.batchsize)
                val_loader = tud.DataLoader(
                    cvsets[i], batch_size=opt.batchsize)
                epoch = cluster + i
                logging.info('Epoch {} start...'.format(epoch+1))
                print('Epoch {} start...'.format(epoch+1))
                loss, _ = train(epoch, train_loader, model,
                                criterion, optimizer, device)
                logging.info('train loss: {}'.format(loss))
                print('train loss: {}'.format(loss))
                loss, acc = val(epoch, val_loader, model,
                                criterion, optimizer, device)
                logging.info('validation acc: {}, loss: {}'.format(acc, loss))
                print('validation acc: {}, loss: {}'.format(acc, loss))
                avg_val_loss += loss
            # calculate avg val loss for early stopper
            avg_val_loss /= len(cvsets)
            logging.info('avg validation loss: {}'.format(avg_val_loss))
            print('avg validation loss: {}'.format(avg_val_loss))
            decision = early_stopper(avg_val_loss)

            logging.info('Testing start...')
            print('Testing start...')
            loss, t_acc = val(epoch, test_loader, model,
                            criterion, optimizer, device)
            logging.info('test acc: {}, loss: {}'.format(t_acc, loss))
            print('test acc: {}, loss: {}'.format(t_acc, loss))

            if decision == 'save':
                saveModel(epoch+1, model, optimizer, os.path.join('exps',
                                                                  opt.exp, 'model_epoch{}.pth'.format(epoch+1)))
                logging.info('model saved.')
                print('model saved.')
            elif decision == 'stop':
                logging.info('early stopped!')
                print('early stopped!')
                break

    else:
        dataset = EEGDataset(os.path.join(opt.datasetPath, 'test'),
                             opt.signalType, opt.freq, opt.winsize, opt.stride, 'test', opt.E, opt.outclass)
        test_loader = tud.DataLoader(dataset, batch_size=opt.batchsize)
        logging.info('Testing start...')
        print('Testing start...')
        loss, acc = val(0, test_loader, model, criterion, optimizer, device)
        print('test acc: {}, loss: {}'.format(acc, loss))
        logging.info('test acc: {}, loss: {}'.format(acc, loss))


main(opt.isTest)
