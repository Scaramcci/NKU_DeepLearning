import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from models import Generator, Discriminator
import os
import numpy as np
from utils import show_imgs

def train_gan(num_epochs=50, batch_size=128, lr=0.0002, beta1=0.5, save_interval=10):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建保存目录
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('images'):
        os.makedirs('images')

    # 数据加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.FashionMNIST(
        root='../FashionMNIST/',
        transform=transform,
        download=True
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    # 初始化模型
    G = Generator().to(device)
    D = Discriminator().to(device)

    # 优化器
    optimizerD = torch.optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = torch.optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))

    # 损失函数
    criterion = nn.BCELoss()

    # 记录损失
    G_losses = []
    D_losses = []

    # 固定噪声用于可视化
    fixed_noise = torch.randn(64, 100, device=device)

    print("开始训练...")
    
    for epoch in range(num_epochs):
        for i, (real_imgs, _) in enumerate(dataloader):
            batch_size = real_imgs.size(0)
            
            # 准备标签
            real_label = torch.ones(batch_size, 1).to(device)
            fake_label = torch.zeros(batch_size, 1).to(device)
            
            # 训练判别器
            D.zero_grad()
            real_imgs = real_imgs.to(device)
            
            # 真实图像的损失
            output_real = D(real_imgs)
            d_loss_real = criterion(output_real, real_label)
            
            # 生成图像的损失
            noise = torch.randn(batch_size, 100, device=device)
            fake_imgs = G(noise)
            output_fake = D(fake_imgs.detach())
            d_loss_fake = criterion(output_fake, fake_label)
            
            # 总判别器损失
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizerD.step()
            
            # 训练生成器
            G.zero_grad()
            output_fake = D(fake_imgs)
            g_loss = criterion(output_fake, real_label)
            g_loss.backward()
            optimizerG.step()
            
            # 记录损失
            G_losses.append(g_loss.item())
            D_losses.append(d_loss.item())
            
            if i % 100 == 0:
                print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] '
                      f'Loss_D: {d_loss.item():.4f} Loss_G: {g_loss.item():.4f}')
                
        # 每个epoch结束时保存生成的图像
        with torch.no_grad():
            fake = G(fixed_noise)
            show_imgs(fake, f'Epoch {epoch}', f'images/fake_epoch_{epoch}.png')
            
        # 定期保存模型
        if (epoch + 1) % save_interval == 0:
            torch.save(G.state_dict(), f'models/generator_epoch_{epoch+1}.pth')
            torch.save(D.state_dict(), f'models/discriminator_epoch_{epoch+1}.pth')
    
    # 保存最终模型
    torch.save(G.state_dict(), 'models/generator_final.pth')
    torch.save(D.state_dict(), 'models/discriminator_final.pth')
    
    # 绘制损失曲线
    plt.figure(figsize=(10,5))
    plt.title("生成器和判别器的Loss")
    plt.plot(G_losses, label="Generator", alpha=0.5)
    plt.plot(D_losses, label="Discriminator", alpha=0.5)
    plt.xlabel("迭代次数")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('images/loss_curves.png')
    plt.show()
    
    return G, D

if __name__ == "__main__":
    G, D = train_gan()
