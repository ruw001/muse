from socket import *
import numpy as np
import argparse
import time
import torch
from resnet import ResNet, BasicBlock, Bottleneck
from resnet_lstm import resnet18_lstm, resnet101_lstm
from scipy import signal as scisig


def extract_band(data, sf):
    fft_vals = np.absolute(np.fft.rfft(data))
    fft_freq = np.fft.rfftfreq(data.shape[1], 1/sf)

    eeg_bands = [(4, 8), (8, 12)]
    res = []

    for band in eeg_bands:
        freq_ix = np.where((fft_freq >= band[0]) &
                           (fft_freq <= band[1]))[0]
        res.append(fft_vals[:, freq_ix])
    return np.concatenate(res, axis=0)

def stft_psd_extract(freq, data):
        '''
            data.shape: 30*256 x 4 
        '''
        psds = []
        for j in range(4):
            f, t, zxx = scisig.stft(data[:,j], fs=freq, nperseg=freq, nfft=freq, axis=0)
            zxx = zxx[3:56, :]
            psds.append(np.abs(zxx))
        return np.array(psds)

def send_from(arr, dest):
    view = memoryview(arr).cast('B')
    while len(view):
        nsent = dest.send(view)
        view = view[nsent:]


def recv_into(arr, source):
    view = memoryview(arr).cast('B')
    timeout = 20
    ts_start = 0
    started = False
    total = 0
    while len(view):
        nrecv = source.recv_into(view)
        if nrecv == 0:
            if not started:
                started = True
                ts_start = time.time()
            else:
                total = time.time() - ts_start
                if total >= timeout:
                    return 1
        else:
            started = False
            total = 0
        view = view[nrecv:]
    return 0

parser = argparse.ArgumentParser(description='Input for EEG AugCog system')
parser.add_argument('-winsize', type=int, default=30, help='window size (s)')
parser.add_argument('-freq', type=int, default=256,
                    help='frequency of signal (Hz)')
parser.add_argument('-modelPath', default='',
                    help='path for loading the model')
parser.add_argument('-useGPU', action='store_true', help='if use GPU or not')
parser.add_argument('-gpuids', nargs='+', type=int,
                    default=[0], help='GPU IDs')
parser.add_argument('-numClass', type=int, default=2, help='#output classes')
parser.add_argument('-E', default='P',
                    help='extract freq bands and use FFT (E) or use psd (P)')

opt = parser.parse_args()
HOST = ''
PORT = 25000
freq = opt.freq
window = np.zeros((opt.winsize * freq, 4))

if opt.E == 'P':
    model = resnet18_lstm(4, opt.numClass)
else:
    model = ResNet(8 if opt.E == 'E' else 4, BasicBlock, [
               2, 2, 2, 2], num_classes=opt.numClass, prob='clf')
if opt.useGPU:
    device = 'cuda:{}'.format(opt.gpuids[0])
    model = model.to(device, dtype=torch.float)
    model = torch.nn.DataParallel(model, device_ids=opt.gpuids)
else:
    device = 'cpu'
    model = model.float()

print('Loading model parameters from {}...'.format(opt.modelPath))
checkpoint = torch.load(opt.modelPath)
model.load_state_dict(checkpoint['state_dict'])
print('Model loaded!')

with socket(AF_INET, SOCK_STREAM) as server:
    server.bind((HOST, PORT))
    count = 0
    server.listen()
    print('Waiting for client...')
    conn, addr = server.accept()
    while True:
        try:
            status = recv_into(window, conn)
            if status == 1:
                print('timeout!')
                conn.close()
                conn, addr = server.accept()
                print('accepted!')
                continue
            print('data received!' + str(count))
            # permute & extract features
            if opt.E == 'E':
                data = np.transpose(window) # c x l
                data = extract_band(data, freq)  # c' = c * len(bands), c' x l
                data = np.expand_dims(data, axis=0) # 1 x c' x l'
            elif opt.E == 'P':
                data = stft_psd_extract(freq, data)
                data = np.expand_dims(data, axis=0) # 1 x c x h x w
                data = np.transpose(data, (0, 1, 3, 2)) # 1 x c x w x h
            else:
                data = np.transpose(window) # c x l
                data = np.expand_dims(data, axis=0) # 1 x c x l
            data = torch.from_numpy(data)
            print('predicting...')
            if device == 'cpu':
                data = data.float()
            else:
                data = data.to(device, dtype=torch.float)
            output = model(data)
            _, pred = output.max(1)
            result = pred[0].item()
            print(result)
            conn.send(bytes(str(result), 'utf-8'))
            print('result sent!', result, 'count: ' + str(count))
            count += 1
        except Exception as e:
            print(e)

