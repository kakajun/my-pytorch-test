import torch
import torchvision
import torchvision.transforms as transforms
import os

# 定义数据转换：转换为 Tensor 并归一化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 创建 data 目录（如果不存在）
if not os.path.exists('./data'):
    os.makedirs('./data')

print("开始下载 MNIST 数据集...")

# 下载训练集
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)

# 下载测试集
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)

print(f"下载完成！")
print(f"训练集大小: {len(trainset)}")
print(f"测试集大小: {len(testset)}")
print(f"数据保存在: {os.path.abspath('./data')}")
