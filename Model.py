
import os
import numpy as np
import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
import pathlib

# Checking device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Transforms
transformer = transforms.Compose([
    transforms.Resize((50, 50)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

# Data Loader
# Path for training and testing directory
train_path = r"C:\Users\Advance Win 10\OneDrive\Desktop\datset proj\NewDataset\NewDataset\Test"
test_path = r"C:\Users\Advance Win 10\OneDrive\Desktop\datset proj\NewDataset\NewDataset\Train"

train_loader = DataLoader(
    torchvision.datasets.ImageFolder(train_path, transform=transformer),
    batch_size=32,
    shuffle=True
)
test_loader = DataLoader(
    torchvision.datasets.ImageFolder(test_path, transform=transformer),
    batch_size=32,
    shuffle=True
)

# Categories
root = pathlib.Path(train_path)
classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])
print(classes)


# CNN Network
class ConvNet(nn.Module):
    def __init__(self, input_dim=3, output_dim=15):
        super(ConvNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.flatten = nn.Flatten()
        
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(256)
        
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3)
        self.batchnorm3 = nn.BatchNorm2d(1024)
        
        self.l1 = nn.Linear(1024*4*4, 512)
        self.l2 = nn.Linear(512, 128)
        self.batchnorm4 = nn.LayerNorm(128)
        self.l3 = nn.Linear(128, output_dim)
        
        self.best_val_acc = 0.0  # Initialize best validation accuracy
        
    def forward(self, input):
        conv = self.conv1(input)
        conv = self.conv2(conv)
        batchnorm = self.relu(self.batchnorm1(conv))
        maxpool = self.maxpool(batchnorm)

        conv = self.conv3(maxpool)
        conv = self.conv4(conv)
        batchnorm = self.relu(self.batchnorm2(conv))
        maxpool = self.maxpool(batchnorm)

        conv = self.conv5(maxpool)
        conv = self.conv6(conv)
        batchnorm = self.relu(self.batchnorm3(conv))
        maxpool = self.maxpool(batchnorm)
        
        flatten = self.flatten(maxpool)
        
        dense_l1 = self.l1(flatten)
        dropout = self.dropout3(dense_l1)
        dense_l2 = self.l2(dropout)
        batchnorm = self.batchnorm4(dense_l2)
        dropout = self.dropout2(batchnorm)
        output = self.l3(dropout)
        
        return output
    


model = ConvNet(input_dim=3, output_dim=15).to(device)

optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
loss_function = nn.CrossEntropyLoss()

num_epochs = 15
# Calculating the size of training and testing images
train_count = len(glob.glob(train_path + '/**/*.png'))
test_count = len(glob.glob(test_path + '/**/*.png'))

print(train_count, test_count)

# Training model and saving best model
best_accuracy = 0.0

for epoch in range(num_epochs):
    # Evaluation and training on training dataset
    model.train()
    train_accuracy = 0.0
    train_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.cpu().data * images.size(0)
        _, prediction = torch.max(outputs.data, 1)

        train_accuracy += int(torch.sum(prediction == labels.data))

    train_accuracy = train_accuracy / train_count
    train_loss = train_loss / train_count

    # Evaluation on testing dataset
    model.eval()

    test_accuracy = 0.0
    for i, (images, labels) in enumerate(test_loader):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)
        test_accuracy += int(torch.sum(prediction == labels.data))

    test_accuracy = test_accuracy / test_count

    print('Epoch: ' + str(epoch) + ' Train Loss: ' + str(train_loss) + ' Train Accuracy: ' + str(
        train_accuracy) + ' Test Accuracy: ' + str(test_accuracy))

    # Save the best model
    if test_accuracy > best_accuracy:
        torch.save(model.state_dict(), 'bestestt.model')
        best_accuracy = test_accuracy
