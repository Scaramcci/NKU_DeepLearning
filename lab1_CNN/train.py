import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from models import Net, ResNet18, DenseNetCIFAR, SEResNet18, Res2NetCIFAR


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def train_model(net, trainloader, testloader, epochs, device, save_path=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    # 用于存储训练过程中的指标
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    print(f"Training on {device}")
    net = net.to(device)
    
    for epoch in range(epochs):
        # 训练阶段
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # 计算accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            running_loss += loss.item()
            if i % 2000 == 1999:
                epoch_loss = running_loss / 2000
                epoch_acc = 100 * correct / total
                print(f'[{epoch + 1}, {i + 1:5d}] train loss: {epoch_loss:.3f}, accuracy: {epoch_acc:.2f}%')
                
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc)
                
                # 验证阶段
                val_loss, val_acc = evaluate_model(net, testloader, criterion, device)
                print(f'[{epoch + 1}, {i + 1:5d}] val loss: {val_loss:.3f}, accuracy: {val_acc:.2f}%')
                
                val_losses.append(val_loss)
                val_accs.append(val_acc)
                
                running_loss = 0.0
                correct = 0
                total = 0
    
    print('Finished Training')
    
    if save_path:
        torch.save(net.state_dict(), save_path)
        print(f'Model saved to {save_path}')
    
    return train_losses, train_accs, val_losses, val_accs


def evaluate_model(net, testloader, criterion, device):
    net.eval()
    val_running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            
            val_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = val_running_loss / len(testloader)
    val_acc = 100 * correct / total
    
    return val_loss, val_acc


def plot_training_curves(train_losses, train_accs, val_losses, val_accs, save_path):
    plt.figure(figsize=(12, 5))
    
    # 绘制loss曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('per 2000 mini-batches')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # 绘制accuracy曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.xlabel('per 2000 mini-batches')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main(model='cnn', epochs=2, batch_size=4, save_path='model.pth'):
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 加载数据
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                           download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                           shuffle=False, num_workers=2)
    
    # 选择模型
    if model == 'cnn':
        net = Net()
    elif model == 'resnet':
        net = ResNet18()
    elif model == 'densenet':
        net = DenseNetCIFAR()
    elif model == 'seresnet':
        net = SEResNet18()
    elif model == 'res2net':
        net = Res2NetCIFAR()
    else:
        raise ValueError(f"不支持的模型类型: {model}，可选项为: cnn, resnet, densenet, seresnet, res2net")
    
    # 训练模型
    train_losses, train_accs, val_losses, val_accs = train_model(
        net, trainloader, testloader, epochs, device, save_path
    )
    
    # 绘制训练曲线
    plot_path = f'image/training_curves_{model}.png'
    plot_training_curves(train_losses, train_accs, val_losses, val_accs, plot_path)
    print(f'Training curves saved to {plot_path}')


if __name__ == '__main__':
    # 如果需要从命令行运行，可以在这里添加参数解析
    # 例如: main(model='resnet', epochs=5, batch_size=8, save_path='resnet_model.pth')
    main(model='cnn', epochs=15, batch_size=16, save_path='cnn_model.pth')
    main(model='resnet', epochs=15, batch_size=16, save_path='resnet_model.pth')
    main(model='densenet', epochs=15, batch_size=16, save_path='densenet_model.pth')
    main(model='seresnet', epochs=15, batch_size=16, save_path='seresnet_model.pth')
    main(model='res2net', epochs=15, batch_size=16, save_path='res2net_model.pth')
