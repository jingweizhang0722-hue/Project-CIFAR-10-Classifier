import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from utils.ResNet import ResNet18

batch_size = 128

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

test_dataset = datasets.CIFAR10(root='./dataset/cifar10/', train=False, download=True, transform=transform_test)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

model = ResNet18()
model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
model.load_state_dict(torch.load('models/model.pt'))

def test():
    model.eval() # 切换到测试模式，BatchNorm 会使用全局均值
    correct = 0
    total = 0
    for data in test_loader:
        images, labels = data[0], data[1]   
        outputs = model(images)
        _, predicted = torch.max(outputs.data, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return 100 * correct / total

acc = test()
print("Model Accuracy :", acc, "%")