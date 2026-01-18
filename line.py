import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. 数据准备
# 随机种子，确保每次运行结果一致
torch.manual_seed(42)

# 生成训练数据
X = torch.randn(100, 2)  # 100 个样本，每个样本 2 个特征
true_w = torch.tensor([2.0, 3.0])  # 假设真实权重
true_b = 4.0  # 偏置项
Y = X @ true_w + true_b + torch.randn(100) * 0.1  # 加入一些噪声

# 打印部分数据
print("部分训练数据 X (前5行):")
print(X[:5])
print("部分目标值 Y (前5个):")
print(Y[:5])

# 2. 定义线性回归模型


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        # 定义一个线性层，输入为2个特征，输出为1个预测值
        self.linear = nn.Linear(2, 1)  # 输入维度2，输出维度1

    def forward(self, x):
        return self.linear(x)  # 前向传播，返回预测结果


# 创建模型实例
model = LinearRegressionModel()

# 3. 定义损失函数与优化器
# 损失函数（均方误差）
criterion = nn.MSELoss()

# 优化器（使用 SGD）
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 学习率设置为0.01

# 4. 训练模型
num_epochs = 1000  # 训练 1000 轮
losses = []  # 记录损失以便画图

print("\n开始训练...")
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式

    # 前向传播
    predictions = model(X)  # 模型输出预测值
    loss = criterion(predictions.squeeze(), Y)  # 计算损失（注意预测值需要压缩为1D）
    losses.append(loss.item())

    # 反向传播
    optimizer.zero_grad()  # 清空之前的梯度
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新模型参数

    # 打印损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 评估模型
print("\n训练完成，评估模型...")
# 查看训练后的权重和偏置
print(f'真实权重: {true_w.numpy()}')
print(f'预测权重: {model.linear.weight.data.numpy().flatten()}')
print(f'真实偏置: {true_b}')
print(f'预测偏置: {model.linear.bias.data.item():.4f}')

# 可视化损失下降过程
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# 在新数据上做预测并可视化
# 为了可视化，我们只展示第一个特征与 Y 的关系（因为是 2D 特征，很难在 2D 平面上完美展示）
model.eval()  # 评估模式
with torch.no_grad():  # 评估时不需要计算梯度
    predictions = model(X).squeeze()

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0].numpy(), Y.numpy(), color='blue',
            label='True values', alpha=0.5)
plt.scatter(X[:, 0].numpy(), predictions.numpy(),
            color='red', label='Predictions', alpha=0.5)
plt.title('Prediction vs True Value (Feature 1)')
plt.xlabel('Feature 1')
plt.ylabel('Y')
plt.legend()

plt.tight_layout()
plt.show()
