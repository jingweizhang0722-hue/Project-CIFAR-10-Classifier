# Project-CIFAR-10-Classifier

本项目基于ResNet18网络模型，通过`Pytorch`训练`Cifar10`分类器，测试集训练准确率达到95%。

#### 1 项目结构

```
    cifar10classifier/
    ├── dataset/            # 存放下载的数据集 #
    ├── runs/               # 生成运行文件，用于tensorboard监督训练 #
    ├── models/             
    │   └── model.pt        # 自动保存/覆盖优化模型 #
    ├── utils/
    │   ├── __pycache__/    
    │   └── ResNet.py       # ResNet网络源文件 #
    ├── main.py             # 项目执行 #
    ├── README.md
    └── use_model.py        # 使用保存的模型 #
```

#### 2 `Cifar10`数据集加载

`Cifar10`数据集由10个类的60000个尺寸为`32x32`的`RGB`彩色图像组成，每个类有6000个图像， 有50000个训练图像和10000个测试图像。

使用`Pytorch`时，通过`torchvision.datasets.CIFAR10()`和`torch.utils.data.DataLoader()`方法加载训练/测试数据集。

```python
train_dataset = datasets.CIFAR10(root='./dataset/cifar10/', train=True, download=True, transform=transform_train)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.CIFAR10(root='./dataset/cifar10/', train=False, download=True, transform=transform_test)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
```

#### 3 `Cifar10`数据集处理

在网络模型的输入上，进行数据集处理。

##### (1)归一化操作

对数据进行`Cifar10`数据集的标准归一化：

```python
torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
```
##### (2)数据增强

为了提高模型的泛化性，一定程度上防止训练过拟合，对训练集数据加以数据增强操作：

```python
torchvision.transforms.RandomCrop(32, padding=4)    # 随机填充裁剪 #
torchvision.transforms.RandomHorizontalFlip()       # 随机水平翻转 #
```

#### 4 `ResNet18`网络模型

通过现有`ResNet18`模型实现分类器训练，将`ResNet18`网络模块的`7x7`卷积层替换为一个`3x3`的卷积，减小该卷积层的步长和填充大小，这样可以尽可能保留原始图像的信息：

```python
model = ResNet18()
model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
```

同时，为了防止过度丢失信息，删去最大池化层：

```python
def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
def forward(self, x):
        return self._forward_impl(x)
```

#### 5 训练策略

通过多次超参数的调试，最终确定训练策略：采用学习率余弦退火方法，初始学习率为0.05，退火周期为120轮（总训练轮数为120），学习率最小值为1e-4

```python
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-4)
```

#### 6 模型保存

训练中，通过每一轮训练对准确率的测试，实现模型保存与覆盖。在模型复用程序中，调用保存的模型：

```python
...
    acc = test()
    if acc > acc_max:
            acc_max = acc
            torch.save(model.state_dict(), 'models/model.pt')
...
```

```python
model.load_state_dict(torch.load('models/model.pt'))
```
#### 7 运用tensorboard实现训练可视化

通过`torch.utils.tensorboard.SummaryWriter()`实现训练过程的可视化。在命令行输入：

```shell
tensorboard --logdir=runs
```

打开网页`localhost:6006`实现训练的监督。