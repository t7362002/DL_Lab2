import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import pandas as pd
from torchvision.io import read_image
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # ==========================
        # TODO 1: build your network
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        
        self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = nn.Linear(in_features=(512 * 8 * 8), out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=10)
        # ==========================


    def forward(self, x):
        # (batch_size, 3, 256, 256)

        # ========================
        # TODO 2: forward the data
        # please write down the output size of each layer
        # example:
        # out = self.relu(self.conv1(x))
        # (batch_size, 64, 256, 256)
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.pool(out)

        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        out = self.pool(out)

        out = self.relu(self.conv5(out))
        out = self.relu(self.conv6(out))
        out = self.relu(self.conv7(out))
        out = self.pool(out)

        out = self.relu(self.conv8(out))
        out = self.relu(self.conv9(out))
        out = self.relu(self.conv10(out))
        out = self.pool(out)

        out = self.relu(self.conv11(out))
        out = self.relu(self.conv12(out))
        out = self.relu(self.conv13(out))
        out = self.pool(out)

        out = torch.flatten(out, 1)

        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        # ========================
        return out


def calc_acc(output, target):
    predicted = torch.max(output, 1)[1]
    num_samples = target.size(0)
    num_correct = (predicted == target).sum().item()
    return num_correct / num_samples


def training(model, device, train_loader, criterion, optimizer):
    # ===============================
    # TODO 3: switch the model to training mode
    model.train()
    # ===============================
    train_acc = 0.0
    train_loss = 0.0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        # =============================================
        # TODO 4: initialize optimizer to zero gradient
        optimizer.zero_grad()
        # =============================================

        output = model(data)
        #print(output)
        #print(target)
        # =================================================
        # TODO 5: loss -> backpropagation -> update weights
        loss = criterion()
        loss = loss(output,target.long())
        loss.backward()
        optimizer.step()
        # =================================================

        train_acc += calc_acc(output, target)
        train_loss += loss.item()

    train_acc /= len(train_loader)
    train_loss /= len(train_loader)

    return train_acc, train_loss


def validation(model, device, valid_loader, criterion):
    # ===============================
    # TODO 6: switch the model to validation mode
    model.eval()
    # ===============================
    valid_acc = 0.0
    valid_loss = 0.0

    # =========================================
    # TODO 7: turn off the gradient calculation
    with torch.no_grad():
    # =========================================
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)

            # ================================
            # TODO 8: calculate accuracy, loss
            loss = criterion()
            loss = loss(output,target)
            valid_acc += calc_acc(output, target.long())
            valid_loss += loss.item()
            # ================================

    valid_acc /= len(valid_loader)
    valid_loss /= len(valid_loader)

    return valid_acc, valid_loss


def main():
    torch.cuda.empty_cache()
    # ==================
    # TODO 9: set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # ==================


    # ========================
    # TODO 10: hyperparameters
    # you can add your parameters here
    LEARNING_RATE = 0.05
    BATCH_SIZE = 25
    EPOCHS = 10
    TRAIN_DATA_PATH = "./data/train/"
    VALID_DATA_PATH = "./data/valid/"
    MODEL_PATH = "model.pt"

    # ========================


    # ===================
    # TODO 11: transforms
    train_transform = transforms.Compose([
        # may be adding some data augmentations?
        #transforms.Resize(256),
        #transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    # ===================
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])


    # =================
    # TODO 12: set up datasets
    # hint: ImageFolder?
    train_data = torchvision.datasets.ImageFolder(TRAIN_DATA_PATH,transform=train_transform)
    valid_data = torchvision.datasets.ImageFolder(VALID_DATA_PATH,transform=valid_transform)
    
    #train_data = datasets.MNIST(TRAIN_DATA_PATH, train=True, download=False, transform=train_transform)
    #train_data = CustomImageDataset(TRAIN_DATA_PATH)
    #valid_data = datasets.MNIST(VALID_DATA_PATH, train=False, download=False, transform=valid_transform)
    # =================


    # ============================
    # TODO 13 : set up dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=int(BATCH_SIZE*9/10))
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=int(BATCH_SIZE/10))
    # ============================

    # build model, criterion and optimizer
    model = Net().to(device).train()
    # ================================
    # TODO 14: criterion and optimizer
    criterion = nn.CrossEntropyLoss
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    # ================================


    # training and validation
    train_acc = [0.0] * EPOCHS
    train_loss = [0.0] * EPOCHS
    valid_acc = [0.0] * EPOCHS
    valid_loss = [0.0] * EPOCHS

    print('Start training...')
    for epoch in range(EPOCHS):
        print(f'epoch {epoch} start...')

        train_acc[epoch], train_loss[epoch] = training(model, device, train_loader, criterion, optimizer)
        valid_acc[epoch], valid_loss[epoch] = validation(model, device, valid_loader, criterion)

        print(f'epoch={epoch} train_acc={train_acc[epoch]} train_loss={train_loss[epoch]} valid_acc={valid_acc[epoch]} valid_loss={valid_loss[epoch]}')
    print('Training finished')


    # ==================================
    # TODO 15: save the model parameters
    torch.save(model.state_dict(), MODEL_PATH)
    # ==================================


    # ========================================
    # TODO 16: draw accuracy and loss pictures
    # lab2_teamXX_accuracy.png, lab2_teamXX_loss.png
    # hint: plt.plot
    x = np.linspace(0, EPOCHS-1, EPOCHS)
    plt.plot(x,train_acc)
    plt.plot(x,valid_acc)
    plt.show()
    plt.plot(x,train_loss)
    plt.plot(x,valid_loss)
    plt.show()

    # =========================================


if __name__ == '__main__':
    main()