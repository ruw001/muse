import torch.nn as nn

class EEG_CNN(nn.Module):
    def __init__(self, winsize, numout):
        super(EEG_CNN, self).__init__()
        conv_st2 = []
        outsize = winsize
        k = 4
        st = 2
        for i in range(3):
            conv_st2.append(nn.Conv1d(4, 4, 4, 2))
            outsize = int((outsize - k)/st) + 1
        self.conv_st2 = nn.ModuleList(conv_st2)
        conv_st4 = []
        k = 8
        st = 4
        for i in range(2):
            conv_st4.append(nn.Conv1d(4, 4, 8, 4))
            outsize = int((outsize - k)/st) + 1
        self.conv_st4 = nn.ModuleList(conv_st4)
        self.fc_1 = nn.Linear(outsize*4, 64)
        self.fc_2 = nn.Linear(64, 32)
        self.fc_3 = nn.Linear(32, numout)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        out = x
        for i in range(len(self.conv_st2)):
            out = self.conv_st2[i](out)
        for i in range(len(self.conv_st4)):
            out = self.conv_st4[i](out)
        out = out.view(out.shape[0], -1)
        out = self.fc_1(out)
        out = self.fc_2(out)
        out = self.fc_3(out)
        return self.softmax(out)

# eeg_cnn = EEG_CNN(5*256, 5)
# print(eeg_cnn)


