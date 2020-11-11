import sys
import os
import torch
import numpy as np
import torch.utils.data as tud
import h5py
from tqdm import tqdm
from scipy import stats
from scipy import signal as scisig

class EEGDataset(tud.Dataset):
    def __init__(self, path, type_, freq, winsize, stride, mode, extract, outClass):
        self.path = path
        self.type = type_
        self.freq = freq
        self.winsize = winsize
        self.stride = stride
        self.mode = mode
        self.extract = extract # N: none, E: extract, P: PSD
        self.label_dict = {}
        for i in range(len(outClass)):
            self.label_dict[outClass[i]] = i
        self.hf_name = "{}_{}_{}_{}_{}_{}.h5".format(mode, type_, winsize, stride, len(outClass), self.extract)
        
        if not self.hf_name in os.listdir(path):
            files = [f for f in os.listdir(path) if self.type in f and f[0] != '.' and 'h5' not in f]
            data = []
            labels = []
            print('Generating dataset...')
            for f in tqdm(files):
                label = int(f.split('_')[1][1:])-1 # stop using '_' in userid!
                if label not in outClass:
                    continue
                # new label
                label = self.label_dict[label]
                signal = []
                with open(os.path.join(path, f), 'r') as infile:
                    lines = infile.readlines()
                    for l in lines:
                        entries = l.split(',')
                        electrodes = [float(e) for e in entries[1:5]] # need to change if data format is different
                        signal.append(electrodes)
                signal = np.array(signal)
                if self.extract == 'N':
                    signal = stats.zscore(signal, axis=0)
                
                ws = int(self.winsize * self.freq)
                st = int(self.stride * self.freq)

                for i in range(0, signal.shape[0], st):
                    labels.append(label)
                    if i+ws > signal.shape[0]:
                        if self.extract == 'P':
                            data.append(self.stft_psd_extract(self.freq, signal[-ws:]))
                        else:
                            data.append(signal[-ws:])
                        break
                    else:
                        if self.extract == 'P':
                            data.append(self.stft_psd_extract(self.freq, signal[i:i+ws]))
                        else:
                            data.append(signal[i:i+ws])
            
            if self.extract == 'P':
                data = np.transpose(np.array(data), (0, 1, 3, 2))  # n,c,w,h
            else:
                data = np.transpose(np.array(data), (0, 2, 1))  # n,c,l
            labels = np.array(labels)
            assert data.shape[0] == labels.shape[0]

            if self.extract == 'E':
                new_data = []
                print('Extracting theta and alpha bands...')
                for i in range(data.shape[0]):
                    res = self.extract_band(data[i,:,:], self.freq) # c * len(bands)
                    new_data.append(res)
                # print(new_data[0].shape)
                data = np.array(new_data)
            # print(data.shape)

            # shuffle dataset
            if self.mode == 'train' and self.extract != 'P':
                indices = np.arange(data.shape[0])
                np.random.shuffle(indices)

                data = data[indices]
                labels = labels[indices]

            print('Creating h5 files...')
            hf = h5py.File(os.path.join(path, self.hf_name), 'w')
            hf.create_dataset(self.mode, data=data)
            hf.create_dataset(self.mode + '_labels', data=labels)
            hf.close()

        self.gt = h5py.File(os.path.join(path, self.hf_name), 'r')
        print('Dataset loaded!')

    def stft_psd_extract(self, freq, data):
        '''
            data.shape: 30*256 x 4 
        '''
        psds = []
        for j in range(4):
            f, t, zxx = scisig.stft(data[:,j], fs=freq, nperseg=freq, nfft=freq, axis=0)
            zxx = zxx[3:56, :]
            psds.append(np.abs(zxx))
        return np.array(psds)

    def extract_band(self, data, sf):
        fft_vals = np.absolute(np.fft.rfft(data))
        fft_freq = np.fft.rfftfreq(data.shape[1], 1/sf)

        # eeg_bands = [(4,8), (8,12), (13,17), (18,22), (23,27), (28,32)]
        eeg_bands = [(4, 8), (8, 12)]
        res = []

        for band in eeg_bands:
            freq_ix = np.where((fft_freq >= band[0]) &
                               (fft_freq <= band[1]))[0]
            res.append(fft_vals[:,freq_ix])
        return np.concatenate(res, axis=0)

    def __len__(self):
        return len(self.gt[self.mode])
    
    def __getitem__(self, index):
        return self.gt[self.mode][index], self.gt[self.mode+'_labels'][index]

# d = EEGDataset('data/ru0802/1', 'EEG', 256, 15, 0.1, 'train', True, [0,4])
# print(d[0][0].shape)

