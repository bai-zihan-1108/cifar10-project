# CIFAR-10 图像分类

基于 PyTorch 的 CIFAR-10 图像分类项目，复现了 [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)，并按要求训练了 ResNet18 模型，使用外部图片进行额外测试。

## 项目结构

```text
cifar10-project/
├── code/                   # 完整项目代码
│   ├── main.py
│   ├── models/
│   ├── utils.py
│   ├── test_simpledla.py   # SimpleDLA 外部图片测试脚本
│   ├── test_resnet18.py    # ResNet18 外部图片测试脚本
│   ├── checkpoint/         # 模型保存目录（训练后生成）
│   ├── data/               # CIFAR-10 数据集（自动下载）
│   └── test_images/        # 外部测试图片
├── report/
│   ├── CIFAR-10 图像分类项目结果报告.pdf     # 项目结果报告
│   ├── 学习心得及未来规划.pdf                # 学习心得及未来规划
└── README.md               # 本文件
```


## 环境要求
Python 3.8+  
PyTorch 2.4.1  
torchvision  
numpy, pillow

推荐使用 Conda 创建虚拟环境：
```
conda create -n pytorch-cifar python=3.8
conda activate pytorch-cifar
pip install torch torchvision
```

## 运行方式
### 准备数据集
程序首次运行时会自动下载 CIFAR-10 数据集。若下载缓慢，可手动下载 [cifar-10-python.tar.gz](https://mirrors.tuna.tsinghua.edu.cn/pytorch/cifar-10-python.tar.gz) 并放入 code/data/ 目录（无需解压）。

### 训练模型
#### 1. 进入代码目录
```
cd code
```
#### 2. 选择要训练的模型
打开 main.py，找到模型定义部分，修改如下：
```
# 训练 ResNet18（考核要求）
net = ResNet18()        # ✅ 使用这一行
# net = SimpleDLA()     # ❌ 注释掉这一行
（如果要训练 SimpleDLA，则反过来注释/取消注释）
```
#### 3. 启动训练
```
python main.py
```
训练过程中，每次测试集准确率创新高时，模型会自动保存到 checkpoint/ckpt.pth。
#### 4. 恢复中断的训练
如果训练意外中断（如电脑休眠、关机），使用以下命令继续：
```
python main.py --resume
```
#### 5. 恢复中断的训练测试外部图片
将待测试的图片放入 code/test_images/ 文件夹，然后运行对应脚本：
```
python test_simpledla.py   # 使用 SimpleDLA 模型
python test_resnet18.py    # 使用 ResNet18 模型
```

## 实验结果
| 模型 | CIFAR-10测试集准确率 | 外部图片测试准确率 |
| --- | --- | --- |
| SimpleDLA | 86.63% |约 89% |
| ResNet18 | 95.48% | 约 99%|

## 常见问题
Windows 下 stty 报错：已修改 utils.py 中的进度条函数，适配 Windows 终端。  
数据集下载慢：使用清华镜像手动下载后放入 code/data/。  
训练中断：使用 --resume 参数恢复训练。
