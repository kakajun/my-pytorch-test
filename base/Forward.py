import torch
import torch.nn as nn
import torch.optim as optim

# 1. 定义一个简单的全连接神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # 简单两层：2 -> 2 -> 1
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 激活函数，增加非线性
        x = self.fc2(x)
        return x

# 2. 准备一点“假数据”
# 假设任务是简单的加法：输入 [a, b]，目标输出 a + b
# 输入：3个样本，每个样本2个特征
inputs = torch.tensor([[1.0, 2.0],
                       [2.0, 3.0],
                       [3.0, 4.0]])

# 目标：对应的真实结果
targets = torch.tensor([[3.0],
                        [5.0],
                        [7.0]])

print(f"输入数据:\n{inputs}")
print(f"目标结果:\n{targets}\n")

# 3. 创建网络实例、损失函数和优化器
model = SimpleNN()

# 随机输入
x = torch.randn(1, 2)

# 前向传播
output = model(x)
print(output)

# 定义损失函数（例如均方误差 MSE）
criterion = nn.MSELoss()

# 假设目标值为 1
target = torch.randn(1, 1)

# 计算损失
loss = criterion(output, target)
print(loss)