import os
import torchvision.transforms as transforms
from PIL import Image

# 定义数据预处理的流水线
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 将图像调整为 128x128
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                         0.229, 0.224, 0.225])  # 标准化
])

# 获取当前脚本所在的目录，构建绝对路径
# 这样无论你在哪里运行脚本，都能找到同级目录下的图片
current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_dir, 'image.jpg')

# 加载图像
image = Image.open(image_path)

# 应用预处理
image_tensor = transform(image)
print(image_tensor.shape)  # 输出张量的形状
