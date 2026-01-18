import numpy as np
import torch
import matplotlib.pyplot as plt
from self import MyDataset  # 导入刚才定义的 MyDataset 类

# 1. 准备更丰富的数据，方便展示效果
# 我们生成 20 个随机点，而不是原来的 4 个点
torch.manual_seed(42)  # 固定随机种子，保证每次生成一样的图

# 生成两组不同的数据，模拟两类（比如：身高体重数据）
# 第一类（Class 0）：分布在左下角 (2, 2) 附近
class0_x = torch.randn(10, 2) + 2
class0_y = torch.zeros(10)

# 第二类（Class 1）：分布在右上角 (6, 6) 附近
class1_x = torch.randn(10, 2) + 6
class1_y = torch.ones(10)

# 合并数据
X_data = torch.cat([class0_x, class1_x], dim=0).numpy()
Y_data = torch.cat([class0_y, class1_y], dim=0).numpy()

# 2. 创建数据集实例
dataset = MyDataset(X_data, Y_data)

# 3. 从数据集中提取数据进行可视化
# 我们模拟从 dataset 中读取数据的过程
all_points = []
all_labels = []

# 遍历整个数据集
for i in range(len(dataset)):
    x, y = dataset[i]
    all_points.append(x.numpy())
    all_labels.append(y.item())

# 转换为方便画图的格式
all_points = np.array(all_points)
all_labels = np.array(all_labels)

# 4. 画图
plt.figure(figsize=(8, 6))

# 画出标签为 0 的点（蓝色圆圈）
plt.scatter(all_points[all_labels == 0][:, 0],
            all_points[all_labels == 0][:, 1],
            color='blue', label='Class 0', s=100, alpha=0.7)

# 画出标签为 1 的点（红色对勾）
# Matplotlib 的 marker 不直接支持 unicode 字符 '✔'，但可以使用 TeX 语法 $\checkmark$
plt.scatter(all_points[all_labels == 1][:, 0],
            all_points[all_labels == 1][:, 1],
            color='red', marker=r'$\checkmark$', label='Class 1', s=100, linewidth=2)

plt.title('Custom Dataset Visualization')
plt.xlabel('Feature 1 (e.g. Height)')
plt.ylabel('Feature 2 (e.g. Weight)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# 给每个点标上索引，方便你对应数据看
for i in range(len(dataset)):
    plt.text(all_points[i, 0]+0.1, all_points[i, 1] +
             0.1, f'Idx:{i}', fontsize=8)

plt.show()
