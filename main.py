import torch
import torch.utils.data as tud
from model import EEG_CNN
from dataset import EEGDataset
from train import train, val

def main():
    model = EEG_CNN(2*256, 5)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.002)
    dataset = EEGDataset('data/dataset_7_12_4', 'EEG', 256, 2, 0.5, 'train')
    train_len = int(len(dataset)*0.7)
    trainset, valset = tud.random_split(dataset, [train_len, len(dataset) - train_len])

    train_loader = tud.DataLoader(trainset, batch_size=10)
    val_loader = tud.DataLoader(valset, batch_size=10)

    for epoch in range(100):
        train(epoch, train_loader, model, criterion, optimizer)
        if epoch % 5 == 0:
            val(epoch, val_loader, model, criterion, optimizer)
    
main()
