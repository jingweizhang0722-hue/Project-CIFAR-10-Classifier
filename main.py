import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from utils.ResNet import ResNet18
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

EPOCH=120
batch_size = 128

# prepare dataset
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

train_dataset = datasets.CIFAR10(root='./dataset/cifar10/', train=True, download=True, transform=transform_train)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.CIFAR10(root='./dataset/cifar10/', train=False, download=True, transform=transform_test)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

# train & test
def train(epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data[0].to(device), data[1].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
 
        running_loss += loss.item()
        if batch_idx % 100 == 99:
            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + batch_idx)
            running_loss = 0.0

def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

if __name__ == '__main__':
    writer = SummaryWriter('runs/cifar10_experiment')

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = ResNet18()
    model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
    model.to(device)

    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    writer.add_graph(model, dummy_input)
 
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-4)

    print("Training for %d epochs..." % EPOCH)
    acc_list = []
    acc_max = 0
    for epoch in range(1, EPOCH + 1):
        train(epoch)
        acc = test()
        writer.add_scalar('Accuracy/test', acc, epoch)
        acc_list.append(acc)
        if acc > acc_max:
            acc_max = acc
            torch.save(model.state_dict(), 'models/model.pt')
        scheduler.step()

    epoch = np.arange(1, EPOCH + 1, 1)
    acc_list = np.array(acc_list)
    plt.plot(epoch, acc_list)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()

writer.close()
