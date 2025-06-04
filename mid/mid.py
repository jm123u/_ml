import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# 数据预处理
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 简单数据增强
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
subset, _ = torch.utils.data.random_split(trainset, [5000, len(trainset)-5000])
trainloader = torch.utils.data.DataLoader(subset, batch_size=32, shuffle=True)

# 定义另一个 CNN 模型
class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        x = self.conv_block(x)
        return self.classifier(x)

# 初始化模型与训练参数
model = CustomCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 训练模型
for epoch in range(3):
    total_loss = 0
    for images, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

print("\n训练完成：模型可以预测时尚物品的类别")

# Fashion-MNIST 类别标签
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 准备测试数据
testset = datasets.FashionMNIST(root='./data', train=False, download=True, 
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,), (0.5,))
                               ]))
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

def predict_single_image():
    """预测单张图片"""
    model.eval()  # 设置为评估模式
    with torch.no_grad():
        # 获取一张测试图片
        images, true_labels = next(iter(testloader))
        
        # 进行预测
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        # 获取预测概率
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence = probabilities[0][predicted].item() * 100
        
        print(f"\n预测结果：")
        print(f"真实标签: {class_names[true_labels[0]]}")
        print(f"预测标签: {class_names[predicted[0]]}")
        print(f"预测置信度: {confidence:.2f}%")
        
        return images[0], true_labels[0], predicted[0], confidence

def evaluate_model():
    """评估模型整体准确率"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False):
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"\n模型在测试集上的准确率: {accuracy:.2f}%")
    return accuracy

# 进行预测演示
print("\n=== 预测演示 ===")
for i in range(3):
    print(f"\n--- 第{i+1}次预测 ---")
    predict_single_image()

# 评估模型性能
evaluate_model()

# 自定义预测函数
def predict_custom_image(image_tensor):
    """
    预测自定义图片
    参数: image_tensor - 形状为 (1, 28, 28) 的张量
    """
    model.eval()
    with torch.no_grad():
        # 确保输入形状正确
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)  # 添加批次维度
        
        # 标准化（如果还没有标准化）
        image_tensor = (image_tensor - 0.5) / 0.5
        
        # 预测
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item() * 100
        
        return class_names[predicted_class], confidence