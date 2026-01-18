import torch

# 创建张量
x = torch.tensor([1, 2, 3])
y = torch.zeros(2, 3)

# 数学运算
z = torch.add(x, 1)  # 逐元素加 1
print(z)

# 索引和切片
mask = x > 1
selected = torch.masked_select(x, mask)
print(selected)

# 设备管理
if torch.cuda.is_available():
    device = torch.device('cuda')
    x = x.to(device)
    print(x.device)
