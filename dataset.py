import sys
import os
import torch
import numpy as np
import torch.utils.data as tud
import h5py
from tqdm import tqdm

class EEGDataset(tud.Dataset):
    def __init__(self, path, type_, freq, winsize, stride, mode):
        self.path = path
        self.type = type_
        self.freq = freq
        self.winsize = winsize
        self.stride = stride
        self.mode = mode
        self.hf_name = "{}_{}_{}_{}.h5".format(mode, type_, winsize, stride)
        
        if not self.hf_name in os.listdir(path):
            files = [f for f in os.listdir(path) if self.type in f and f[0] != '.']
            data = []
            labels = []
            print('Generating dataset...')
            for f in tqdm(files):
                label = int(f.split('_')[1][1:])-1
                signal = []
                with open(os.path.join(path, f), 'r') as infile:
                    lines = infile.readlines()
                    for l in lines:
                        entries = l.split(',')
                        electrodes = [float(e) for e in entries[1:5]] # need to change if data format is different
                        signal.append(electrodes)
                signal = np.array(signal)
                ws = int(self.winsize * self.freq)
                st = int(self.stride * self.freq)

                for i in range(0, signal.shape[0], st):
                    if i+ws > signal.shape[0]:
                        data.append(signal[-ws:])
                    else:
                        data.append(signal[i:i+ws])
                    labels.append(label)
            
            data = np.transpose(np.array(data), (0,2,1))
            labels = np.array(labels)
            assert data.shape[0] == labels.shape[0]

            indices = np.arange(data.shape[0])
            np.random.shuffle(indices)

            data = data[indices,:,:]
            labels = labels[indices]

            print('Creating h5 files...')
            hf = h5py.File(os.path.join(path, self.hf_name), 'w')
            hf.create_dataset(self.mode, data=data)
            hf.create_dataset(self.mode + '_labels', data=labels)
            hf.create_dataset(self.mode + '_len', data=data.shape[0])
            hf.close()

        self.gt = h5py.File(os.path.join(path, self.hf_name), 'r')
        print('Dataset loaded!')

    def __len__(self):
        return len(self.gt[self.mode])
    
    def __getitem__(self, index):
        return self.gt[self.mode][index], self.gt[self.mode+'_labels'][index]

# d = EEGDataset('data/dataset_7_12_4', 'EEG', 256, 2, 0.5, 'train')
# print(len(d))

