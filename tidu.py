import torch
# 创建一个需要梯度的张量
tensor_requires_grad = torch.tensor([1.0, 2.0], requires_grad=True)

# 进行一些操作
tensor_result = tensor_requires_grad ** 2 + 3 * tensor_requires_grad

# 计算梯度
tensor_result.sum().backward()
print(tensor_requires_grad.grad)  # 输出梯度