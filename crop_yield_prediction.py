import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 场景说明：农作物产量预测
# ==========================================
# 假设我们想预测小麦的亩产量（Yield, 单位：kg）
# 影响因素（特征 X）：
# 1. 施肥量 (Fertilizer): 单位 kg/亩
# 2. 降雨量 (Rainfall): 单位 mm
# 3. 日照时间 (Sunshine): 单位 hours
#
# 假设真实关系（上帝视角）：
# Yield = 3.5 * 施肥量 + 0.8 * 降雨量 + 1.2 * 日照时间 + 200 (基础产量)
# ==========================================

print("=== 农作物产量预测模型 (Linear Regression) ===")

# 1. 模拟历史数据 (Data Preparation)
torch.manual_seed(123)

n_samples = 200  # 收集了 200 块农田的数据

# 生成随机特征数据 (X)
# 施肥量: 50-150 kg
fertilizer = torch.rand(n_samples, 1) * 100 + 50
# 降雨量: 300-800 mm
rainfall = torch.rand(n_samples, 1) * 500 + 300
# 日照时间: 1000-2500 hours
sunshine = torch.rand(n_samples, 1) * 1500 + 1000

# 合并成特征矩阵 X [200, 3]
X_raw = torch.cat((fertilizer, rainfall, sunshine), dim=1)

# === 关键步骤：数据归一化 (Data Normalization) ===
# 因为特征数值差异很大（比如施肥是100左右，日照是2000左右），
# 直接训练会导致梯度爆炸或者收敛极慢（偏置项更新不动）。
# 我们将数据标准化到均值为0，方差为1的分布。
X_mean = X_raw.mean(dim=0)
X_std = X_raw.std(dim=0)
X = (X_raw - X_mean) / X_std

# 定义真实的权重和偏置 (用于生成标签)
# 注意：这里的真实权重是基于原始数据的，模型训练后学到的权重会是基于归一化数据的，
# 所以最后打印出来的权重数值会不一样，但预测结果是一样的。
true_w = torch.tensor([3.5, 0.8, 1.2])
true_b = 200.0

# 生成目标产量 (Y) 使用原始数据 X_raw
# 目标值 Y 不需要归一化，当然归一化也可以，这里为了直观就不归一化 Y 了
noise = torch.randn(n_samples) * 20
Y = torch.matmul(X_raw, true_w) + true_b + noise
Y = Y.unsqueeze(1)

print(f"数据准备完毕: {n_samples} 条样本")
print(f"原始特征示例: {X_raw[0].tolist()}")
print(f"归一化特征示例: {X[0].tolist()}")
print(f"对应产量: {Y[0].item():.2f} kg\n")


# 2. 定义模型 (Model Definition)
class CropYieldModel(nn.Module):
    def __init__(self):
        super(CropYieldModel, self).__init__()
        # 输入3个特征，输出1个产量值
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)


model = CropYieldModel()


# 3. 训练配置 (Training Config)
criterion = nn.MSELoss()  # 均方误差
# 归一化后，我们可以使用正常的学习率了！
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

print("开始训练模型...")

# 4. 训练循环 (Training Loop)
num_epochs = 2000  # 2000轮足够了
losses = []

for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X)
    loss = criterion(y_pred, Y)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if (epoch+1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.2f}')

print("训练结束！\n")


# 5. 结果分析与应用 (Analysis & Application)
print("=== 模型分析 ===")
pred_w = model.linear.weight.data[0]
pred_b = model.linear.bias.data[0]

# 注意：因为我们训练的是归一化后的数据，所以这里的权重对应的是归一化后的特征
print(f"真实规律 (基于原始数据): Yield = 3.50 * 施肥 + 0.80 * 降雨 + 1.20 * 日照 + 200")
print(
    f"模型学到 (基于归一化数据): Yield = {pred_w[0]:.2f} * Norm(施肥) + {pred_w[1]:.2f} * Norm(降雨) + {pred_w[2]:.2f} * Norm(日照) + {pred_b:.2f}")

# 6. 实际应用场景演示
print("\n=== 实际应用：预测明年产量 ===")
# 假设明年计划：施肥 120kg, 预计降雨 600mm, 预计日照 2000小时
new_plan_raw = torch.tensor([[120.0, 600.0, 2000.0]])

# === 关键：预测时也要对新数据做同样的归一化 ===
new_plan = (new_plan_raw - X_mean) / X_std

model.eval()
with torch.no_grad():
    predicted_yield = model(new_plan).item()

# 人工算一下理论值
manual_calc = 3.5 * 120 + 0.8 * 600 + 1.2 * 2000 + 200

print(f"明年计划投入:")
print(f"  - 施肥: 120 kg")
print(f"  - 预计降雨: 600 mm")
print(f"  - 预计日照: 2000 hours")
print(f"--------------------------------")
print(f"🤖 AI 预测亩产量: {predicted_yield:.2f} kg")
print(f"📝 理论公式计算: {manual_calc:.2f} kg")
print(f"--------------------------------")
print(f"误差: {abs(predicted_yield - manual_calc):.2f} kg ({(abs(predicted_yield - manual_calc)/manual_calc)*100:.2f}%)")

# 可视化损失
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title('Training Loss (Crop Yield Prediction)')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.grid(True)
plt.show()
