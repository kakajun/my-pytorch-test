"""
是让小学生做 几何题 （非线性关系）。

如果用一把直尺（线性模型）去切蛋糕，你很难一刀把圆心和圆外完美分开。
这通常用来演示 神经网络为什么需要“激活函数”（比如 ReLU 或 Sigmoid）
因为只有有了非线性的能力，神经网络才能学会画“圆圈”这样的复杂边界。
"""
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 生成一些随机数据
n_samples = 100
data = torch.randn(n_samples, 2)  # 生成 100 个二维数据点
labels = (data[:, 0]**2 + data[:, 1]**2 <
          1).float().unsqueeze(1)  # 点在圆内为1，圆外为0

# 可视化数据
plt.scatter(data[:, 0], data[:, 1], c=labels.squeeze(), cmap='coolwarm')
plt.title("Generated Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
