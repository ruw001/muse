import torch.nn as nn


class EEG_CNN(nn.Module):
    def __init__(self, winsize, numout, prob='clf'):
        super(EEG_CNN, self).__init__()
        # input channel must be 4
        conv_st2 = []
        outsize = winsize
        st = 2
        k = 4
        for i in range(3):
            conv_st2.append(nn.Conv1d(4, 4, k, st))
            outsize = int((outsize - k)/st) + 1
        self.conv_st2 = nn.ModuleList(conv_st2)

        conv_st4 = []
        st = 4
        k = 8
        for i in range(2):
            conv_st4.append(nn.Conv1d(4, 4, k, st))
            outsize = int((outsize - k)/st) + 1
        self.conv_st4 = nn.ModuleList(conv_st4)

        st = 8
        k = 16
        self.conv_st8 = nn.Conv1d(4, 8, k, st)
        outsize = int((outsize - k)/st) + 1

        self.relu = nn.ReLU()
        self.drop = nn.Dropout()

        self.fc_1 = nn.Linear(outsize*8, 64)
        self.fc_2 = nn.Linear(64, 32)
        self.fc_3 = nn.Linear(32, numout)
        self.fc_reg = nn.Linear(32, 1)
        self.prob = prob

    def forward(self, x):
        out = x
        for i in range(len(self.conv_st2)):
            out = self.conv_st2[i](out)
            out = self.relu(out)
        for i in range(len(self.conv_st4)):
            out = self.conv_st4[i](out)
            out = self.relu(out)
        out = self.relu(self.conv_st8(out))
        out = out.view(out.shape[0], -1)
        out = self.drop(out)
        out = self.fc_1(out)
        out = self.drop(self.relu(out))
        out = self.fc_2(out)
        if self.prob == 'clf':
            out = self.fc_3(out)
        else:
            out = self.fc_reg(out)
        return out

class Depricated_EEG_CNN(nn.Module):
    def __init__(self, winsize, numout):
        super(Depricated_EEG_CNN, self).__init__()
        conv_st2 = []
        bn_st2 = []
        outsize = winsize
        k = 4
        st = 2
        for i in range(3):
            bn_st2.append(nn.BatchNorm1d(4))
            conv_st2.append(nn.Conv1d(4, 4, 4, 2))
            outsize = int((outsize - k)/st) + 1
        self.conv_st2 = nn.ModuleList(conv_st2)
        self.bn_st2 = nn.ModuleList(bn_st2)
        
        conv_st4 = []
        bn_st4 = []
        k = 8
        st = 4
        for i in range(2):
            bn_st4.append(nn.BatchNorm1d(4))
            conv_st4.append(nn.Conv1d(4, 4, 8, 4))
            outsize = int((outsize - k)/st) + 1
        self.conv_st4 = nn.ModuleList(conv_st4)
        self.bn_st4 = nn.ModuleList(bn_st4)

        self.relu = nn.ReLU()
        self.drop = nn.Dropout()

        self.fc_1 = nn.Linear(outsize*4, 64)
        self.fc_2 = nn.Linear(64, 32)
        self.fc_3 = nn.Linear(32, numout)
    
    def forward(self, x):
        out = x
        for i in range(len(self.conv_st2)):
            out = self.conv_st2[i](out)
            out = self.bn_st2[i](out)
            out = self.relu(out)
        for i in range(len(self.conv_st4)):            
            out = self.conv_st4[i](out)
            out = self.bn_st4[i](out)
            out = self.relu(out)
        out = out.view(out.shape[0], -1)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc_2(out)
        out = self.relu(out)
        out = self.fc_3(out)
        return out

# eeg_cnn = EEG_CNN(5*256, 5)
# print(eeg_cnn)


