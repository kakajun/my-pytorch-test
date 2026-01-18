from self import MyDataset
import sys
import os
from torch.utils.data import DataLoader

# 将当前目录添加到 sys.path 以便导入同目录下的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# 准备示例数据
X_data = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]
Y_data = [0, 1, 0, 1, 0, 1]

# 实例化数据集
dataset = MyDataset(X_data, Y_data)

# 创建 DataLoader 实例，batch_size 设置每次加载的样本数量
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 打印加载的数据
for epoch in range(1):
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        print(f'Batch {batch_idx + 1}:')
        print(f'Inputs: {inputs}')
        print(f'Labels: {labels}')
