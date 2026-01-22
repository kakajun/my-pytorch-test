import torch
import torch.nn as nn
import torch.optim as optim

# --- 0. 固定随机种子 (解决每次结果不一样的问题) ---
# 只要种子一样，每次运行生成的随机数（初始权重）就一样
torch.manual_seed(42)

# 1. 定义一个简单的全连接神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # 简单两层：2 -> 10 -> 1 (增加隐藏层神经元)
        # 增加神经元数量确实会增加计算量，但在这种微型任务上，时间差异肉眼不可见
        # 但它能极大提高模型的稳定性，避免“死神经元”问题
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 1)

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

criterion = nn.MSELoss()  # 均方误差损失：(预测值 - 真实值)^2
optimizer = optim.SGD(model.parameters(), lr=0.01) # SGD优化器，学习率0.01

print("开始训练... (观察 Loss 也就是误差是不是在变小)")
print("-" * 30)

# 4. 简单的训练循环（只跑100次）
for epoch in range(100):
    # --- 前向传播 ---
    outputs = model(inputs)        # 1. 喂数据，算预测值
    loss = criterion(outputs, targets) # 2. 算误差 (Loss)

    # --- 反向传播 ---
    optimizer.zero_grad()          # 3. 清空上一步的梯度（如果不清空会累加）
    loss.backward()                # 4. 算梯度 (反向传播)

    optimizer.step()               # 5. 更新参数 (用梯度调整权重)

    if epoch == 0:
        print("--- 第一次更新后的参数 (fc1.weight) ---")
        print(model.fc1.weight.data)
        print("--- 可以看到：新参数 = 旧参数 - 学习率 * 梯度 ---\n")

    # 每20次打印一下进度
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")

print("-" * 30)
print("训练结束！")

# 5. 验证一下训练成果
with torch.no_grad(): # 推理时不需要算梯度
    test_input = torch.tensor([[7.2, 8.4]]) # 测试一个没见过的数据：7.2+8.4=?
    predicted = model(test_input)
    print(f"测试输入: [7.2, 8.4]")
    print(f"模型预测: {predicted.item():.4f} (理想应该是 15.6)")
