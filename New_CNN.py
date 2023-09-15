import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import read_image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
epochs = 2000
batch_size = 16
learning_rate = 0.001
momentum = 0.2
image_size = (360, 640)



#Data preparation
class CustomImageDataset(Dataset):
    def __init__(self, labels_arr, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.DataFrame(labels_arr)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = torch.tensor(int(self.img_labels.iloc[idx, 1]))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
    
def create_dataset(folder, label):
    files = os.listdir(folder)
    for i in files:
        if i =='.DS_Store':
            files.remove(i)
    files = np.array(files)
    labels_arr = np.zeros(len(files), dtype=np.uint8)
    labels_arr.fill(label)
    labels_arr = np.column_stack((files, labels_arr))

    transform = transforms.Compose([transforms.Resize(image_size)])
    dataset = CustomImageDataset(labels_arr, folder, transform=transform)
    return dataset

#dataloader_low = create_dataset("/Users/gaky/Desktop/efir/low", int(2))
#ataloader_medium = create_dataset("/Users/gaky/Desktop/efir/medium", int(3))
#dataloader_high = create_dataset("/Users/gaky/Desktop/efir/high_small", int(0))
#dataloader_off = create_dataset("/Users/gaky/Desktop/efir/off", int(1))

#dataloader_low_blur = create_dataset("/Users/gaky/Desktop/efir/low blur", int(0))
#dataloader_medium_blur = create_dataset("/Users/gaky/Desktop/efir/medium blur", int(1))
#dataloader_high_blur = create_dataset("/Users/gaky/Desktop/efir/high_small", int(0))
#dataloader_off_blur = create_dataset("/Users/gaky/Desktop/efir/off blur", int(1))

dataloader_on = create_dataset("/Users/gaky/Desktop/efir/on", int(0))
dataloader_off = create_dataset("/Users/gaky/Desktop/efir/off mix", int(1))

dataloader_val_on = create_dataset("/Users/gaky/Desktop/efir/validation on", int(0))
dataloader_val_off = create_dataset("/Users/gaky/Desktop/efir/validation off mix", int(1))


train_full = ConcatDataset([dataloader_on, dataloader_off])

val_on = ConcatDataset([dataloader_val_on])
val_off = ConcatDataset([dataloader_val_off])


dataloader_train = DataLoader(train_full, batch_size=batch_size, shuffle=True)

dataloader_train_full = DataLoader(train_full, batch_size=(len(train_full)), shuffle=True)

dataloader_val_on = DataLoader(val_on, batch_size=(len(dataloader_val_on)), shuffle=True)

dataloader_val_off = DataLoader(val_off, batch_size=(len(dataloader_val_off)), shuffle=True)



class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=1, padding=1)#360x640
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        
        self.conv1_bn = nn.BatchNorm2d(64)
        
        self.pool1 = nn.MaxPool2d(2, stride=2)#180x320
        
        self.pool1_bn = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)#180x320
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        
        self.conv2_bn = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)#180x320
        nn.init.kaiming_normal_(self.conv3.weight, mode='fan_in', nonlinearity='relu')
        
        self.conv3_bn = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)#180x320
        nn.init.kaiming_normal_(self.conv4.weight, mode='fan_in', nonlinearity='relu')
        
        self.conv4_bn = nn.BatchNorm2d(64)
        
        self.pool4 = nn.MaxPool2d(2, stride=2)#90x160
        
        self.pool4_bn = nn.BatchNorm2d(64)
        
        self.conv5 = nn.Conv2d(64, 128, 3, stride=1, padding=1)#90x160
        nn.init.kaiming_normal_(self.conv5.weight, mode='fan_in', nonlinearity='relu')

        self.conv5_bn = nn.BatchNorm2d(128)

        self.conv6 = nn.Conv2d(128, 128, 3, stride=1, padding=1)#90x160
        nn.init.kaiming_normal_(self.conv6.weight, mode='fan_in', nonlinearity='relu')
        
        self.conv6_bn = nn.BatchNorm2d(128)
       
        self.pool6 = nn.MaxPool2d(2, stride=2)#45x80
        
        self.pool6_bn = nn.BatchNorm2d(128)
        
        self.conv7 = nn.Conv2d(128, 128, 3, stride=1, padding=1)#45x80
        nn.init.kaiming_normal_(self.conv7.weight, mode='fan_in', nonlinearity='relu')
        
        self.conv7_bn = nn.BatchNorm2d(128)
        
        self.conv8 = nn.Conv2d(128, 128, 3, stride=1, padding=1)#45x80
        nn.init.kaiming_normal_(self.conv8.weight, mode='fan_in', nonlinearity='relu')
        
        self.conv8_bn = nn.BatchNorm2d(128)
        
        self.pool8 = nn.MaxPool2d(2, stride=2)#23x40
        
        self.pool8_bn = nn.BatchNorm2d(128)
        
        self.conv9 = nn.Conv2d(128, 128, 3, stride=1, padding=1)#23x40
        nn.init.kaiming_normal_(self.conv9.weight, mode='fan_in', nonlinearity='relu')
        
        self.conv9_bn = nn.BatchNorm2d(128)
        
        self.conv10 = nn.Conv2d(128, 128, 3, stride=1, padding=1)#23x40
        nn.init.kaiming_normal_(self.conv10.weight, mode='fan_in', nonlinearity='relu')
        
        self.conv10_bn = nn.BatchNorm2d(128)
        
        self.pool10 = nn.MaxPool2d(2, stride=2)#11x20
        
        self.pool10_bn = nn.BatchNorm2d(128)
    
        self.conv11 = nn.Conv2d(128, 128, 3, stride=1, padding=1)#11x20
        nn.init.kaiming_normal_(self.conv11.weight, mode='fan_in', nonlinearity='relu')
        
        self.conv11_bn = nn.BatchNorm2d(128)
        
        self.conv12 = nn.Conv2d(128, 128, 3, stride=1, padding=1)#11x20
        nn.init.kaiming_normal_(self.conv12.weight, mode='fan_in', nonlinearity='relu')
        
        self.conv12_bn = nn.BatchNorm2d(128)
        
        self.pool12 = nn.AvgPool2d(2, stride=2)#6x10
        
        self.pool12_bn = nn.BatchNorm2d(128)
            
        
        self.fc1 = nn.Linear(5*9*128, 1000)
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        
        self.fc1_bn = nn.BatchNorm1d(1000)
        
        self.fc2 = nn.Linear(1000, 1000)
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        
        self.fc2_bn = nn.BatchNorm1d(1000)
        
        self.fc3 = nn.Linear(1000, 2)
        nn.init.kaiming_normal_(self.fc3.weight, mode='fan_in', nonlinearity='relu')
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = F.relu(x)
        
        x = self.pool1(x)
        x = self.pool1_bn(x)
        
        
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = F.relu(x)
        
        
        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = F.relu(x)
        
        
        x = self.conv4(x)
        x = self.conv4_bn(x)
        x = F.relu(x)
        
        x = self.pool4(x)
        x = self.pool4_bn(x)
        
        
        x = self.conv5(x)
        x = self.conv5_bn(x)
        x = F.relu(x)
        
        x = self.conv6(x)
        x = self.conv6_bn(x)
        x = F.relu(x)
        
        x = self.pool6(x)
        x = self.pool6_bn(x)
        
        
        x = self.conv7(x)
        x = self.conv7_bn(x)
        x = F.relu(x)
        
        
        x = self.conv8(x)
        x = self.conv8_bn(x)
        x = F.relu(x)
        
        x = self.pool8(x)
        x = self.pool8_bn(x)
        
        
        x = self.conv9(x)
        x = self.conv9_bn(x)
        x = F.relu(x)
        
        
        x = self.conv10(x)
        x = self.conv10_bn(x)
        x = F.relu(x)
        
        x = self.pool10(x)
        x = self.pool10_bn(x)
        
    
        x = self.conv11(x)
        x = self.conv11_bn(x)
        x = F.relu(x)
        
            
        x = self.conv12(x)
        x = self.conv12_bn(x)
        x = F.relu(x)
        
        x = self.pool12(x)
        x = self.pool12_bn(x)
        
        
        x = x.view(-1, 5*9*128)   
        
        
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = F.relu(x)
        
        #x = F.dropout(x)
        
        x = self.fc3(x)
        x = F.relu(x)
        return x


model = ConvNet().to(device)
#0,4250
#0,575
#weight = [0,2875, 0,7125]
weights = np.array([4.5, 5.5])

weights = torch.tensor(weights).float()


loss_function = nn.CrossEntropyLoss()
#optim = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)




for epoch in range(epochs):
    for i, (x, y) in enumerate(dataloader_train):

        x = x/255.
        x = x.to(device)
        y = y.to(device)

        model.zero_grad()
        x = F.normalize(x.float())
        out = model(x.float())

        loss = loss_function(out, y.long())
        loss = loss.mean()
        loss.backward()
        optim.step()

        
        
        if epoch % 2 == 0:
            dataset_train = iter(dataloader_train_full)
            x, y = next(dataset_test)
            x = x/255.
            x = x.to(device)
            y = y.to(device)

            with torch.no_grad():
            out = model(x.float())
            cat = torch.argmax(out, dim=1)
            off = (cat == 1).float().mean() 
            on = (cat == 0).float().mean() 

            accuracy = (cat == y.long()).float().mean()
            print(f"off_train: {off}, on_train: {on}, test_accuracy, {accuracy.item()}")
            
            
            val_on = iter(dataloader_val_on)
            x, y = next(val_on)
            x = x/255.
            x = x.to(device)
            y = y.to(device)

            with torch.no_grad():
            out = model(x.float())
            cat = torch.argmax(out, dim=1)

            accuracy = (cat == y.long()).float().mean()
            print(f"val_on: {accuracy}")  
            
            
            val_on = iter(dataloader_val_off)
            x, y = next(val_on)
            x = x/255.
            x = x.to(device)
            y = y.to(device)

            with torch.no_grad():
            out = model(x.float())
            cat = torch.argmax(out, dim=1)

            accuracy = (cat == y.long()).float().mean()
            print(f"val_off: {accuracy}")  
            
        print(epoch, i)
