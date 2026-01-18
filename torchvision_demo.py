import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import time

def main():
    # 1. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 2. 数据准备
    print("准备数据 (CIFAR10)...")
    
    # 定义训练集的变换：数据增强
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # CIFAR10 均值和标准差
    ])

    # 定义测试集的变换：仅标准化
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 下载并加载数据集
    # root='./data' 指定数据存储路径
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # 3. 模型定义
    print("加载预训练 ResNet18 模型...")
    # 使用预训练模型
    model = models.resnet18(pretrained=True)
    
    # 修改最后一层全连接层，适应 CIFAR10 的 10 分类
    # ResNet18 的 fc 层输入特征数是 512
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    
    model = model.to(device)

    # 4. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    # 使用 SGD 优化器，学习率 0.001，动量 0.9
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 5. 训练循环
    print("开始训练...")
    epochs = 5 # 演示目的，只训练 5 轮
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        
        for i, data in enumerate(trainloader, 0):
            # 获取输入
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播 + 反向传播 + 优化
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 统计损失
            running_loss += loss.item()
            if i % 200 == 199:    # 每 200 个 batch 打印一次
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0
        
        end_time = time.time()
        print(f'Epoch {epoch + 1} 完成, 耗时: {end_time - start_time:.2f}秒')

    print('训练完成')

    # 6. 模型评估
    print("在测试集上评估模型...")
    model.eval()
    correct = 0
    total = 0
    
    # 不计算梯度
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'测试集 (10000 张图片) 准确率: {100 * correct / total:.2f}%')

    # 类别准确率详情
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4): # batch_size 可能是 32，这里简单处理最后不足的情况
                if i < len(labels):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

    print("\n各类别准确率:")
    for i in range(10):
        if class_total[i] > 0:
            print(f'{classes[i]:5s} : {100 * class_correct[i] / class_total[i]:.2f}%')

    # 7. 保存模型
    save_path = './cifar_net.pth'
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存至: {save_path}")

if __name__ == '__main__':
    main()
