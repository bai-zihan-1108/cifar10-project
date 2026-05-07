import torch
import torchvision.transforms as transforms
from PIL import Image
import os

# 导入 ResNet18
from models.resnet import ResNet18

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#使用 ResNet18 模型
model = ResNet18().to(device)
checkpoint = torch.load('./checkpoint/ckpt.pth', map_location=device)
model.load_state_dict(checkpoint['net'])
model.eval()
print("ResNet18 模型加载成功！")

# 预处理
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

test_dir = './test_images'
if not os.path.exists(test_dir):
    os.makedirs(test_dir)
    print(f"已创建 {test_dir} 文件夹，请放入几张测试图片后重新运行")
    exit()

for img_name in os.listdir(test_dir):
    if img_name.endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(test_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_tensor.to(device))
            _, predicted = output.max(1)
        
        print(f'{img_name} -> 预测类别: {classes[predicted.item()]}')