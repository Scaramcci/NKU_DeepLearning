import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from improved_cnn_models import Generator, Discriminator
import os
import numpy as np
from utils import show_imgs
import random

def train_gan(num_epochs=50, batch_size=32, lr_g=0.0002, lr_d=0.0001, beta1=0.5, save_interval=5):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建保存目录
    if not os.path.exists('improved_models'):
        os.makedirs('improved_models')
    if not os.path.exists('improved_images'):
        os.makedirs('improved_images')

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

    # 优化器 - 使用不同的学习率
    optimizerD = torch.optim.Adam(D.parameters(), lr=lr_d, betas=(beta1, 0.999))
    optimizerG = torch.optim.Adam(G.parameters(), lr=lr_g, betas=(beta1, 0.999))
    
    # 学习率调度器
    schedulerD = torch.optim.lr_scheduler.ExponentialLR(optimizerD, gamma=0.99)
    schedulerG = torch.optim.lr_scheduler.ExponentialLR(optimizerG, gamma=0.99)
    
    # 损失函数
    criterion = nn.BCELoss()

    # 记录损失
    G_losses = []
    D_losses = []
    D_real_scores = []
    D_fake_scores = []

    # 固定噪声用于可视化
    fixed_noise = torch.randn(64, 100, device=device)

    print("开始训练...")
    print(f"生成器学习率: {lr_g}, 判别器学习率: {lr_d}")
    print(f"批次大小: {batch_size}, 训练轮数: {num_epochs}")
    
    for epoch in range(num_epochs):
        for i, (real_imgs, _) in enumerate(dataloader):
            batch_size = real_imgs.size(0)
            
            # 准备标签 - 使用标签平滑和噪声
            # 真实标签: 0.7-0.9 之间的随机值
            real_label = torch.rand(batch_size, device=device) * 0.2 + 0.7
            # 假标签: 0.0-0.2 之间的随机值
            fake_label = torch.rand(batch_size, device=device) * 0.2
            
            # 训练判别器
            D.zero_grad()
            
            # 添加噪声到真实图像
            real_imgs = real_imgs.to(device)
            noise_factor = 0.05
            real_imgs_noisy = real_imgs + noise_factor * torch.randn_like(real_imgs)
            real_imgs_noisy = torch.clamp(real_imgs_noisy, -1, 1)
            
            # 真实图像的损失
            output_real = D(real_imgs_noisy)
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
            
            # 记录判别器得分
            D_real_scores.append(output_real.mean().item())
            D_fake_scores.append(output_fake.mean().item())
            
            # 训练生成器 - 每个批次都训练生成器
            G.zero_grad()
            # 重新生成fake_imgs以避免梯度问题
            noise = torch.randn(batch_size, 100, device=device)
            fake_imgs = G(noise)
            output_fake = D(fake_imgs)
            
            # 生成器希望判别器将假图像判断为真
            g_loss = criterion(output_fake, torch.ones(batch_size, device=device))
            g_loss.backward()
            optimizerG.step()
            
            # 记录损失
            G_losses.append(g_loss.item())
            D_losses.append(d_loss.item())
            
            if i % 100 == 0:
                print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] '
                      f'Loss_D: {d_loss.item():.4f} Loss_G: {g_loss.item():.4f} '
                      f'D(x): {output_real.mean().item():.4f} D(G(z)): {output_fake.mean().item():.4f} '
                      f'LR_D: {optimizerD.param_groups[0]["lr"]:.6f} LR_G: {optimizerG.param_groups[0]["lr"]:.6f}')
                
        # 每个epoch结束时保存生成的图像
        with torch.no_grad():
            fake = G(fixed_noise)
            show_imgs(fake, f'Epoch {epoch}', f'improved_images/fake_epoch_{epoch}.png')
            
        # 更新学习率
        schedulerD.step()
        schedulerG.step()
        
        # 定期保存模型
        if (epoch + 1) % save_interval == 0:
            torch.save(G.state_dict(), f'improved_models/generator_epoch_{epoch+1}.pth')
            torch.save(D.state_dict(), f'improved_models/discriminator_epoch_{epoch+1}.pth')
            print(f'模型已保存 - Epoch {epoch+1}')
    
    # 保存最终模型
    torch.save(G.state_dict(), 'improved_models/generator_final.pth')
    torch.save(D.state_dict(), 'improved_models/discriminator_final.pth')
    
    # 绘制损失曲线
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.title("Generator and Discriminator Loss")
    plt.plot(G_losses, label="Generator", alpha=0.5)
    plt.plot(D_losses, label="Discriminator", alpha=0.5)
    plt.xlabel("Generations")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.title("Discriminator Scores")
    plt.plot(D_real_scores, label="D(x)", alpha=0.5)
    plt.plot(D_fake_scores, label="D(G(z))", alpha=0.5)
    plt.xlabel("迭代次数")
    plt.ylabel("Score")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('improved_images/training_curves.png')
    plt.show()
    
    return G, D

def analyze_results(G, D):
    """分析训练结果"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 生成一批图像
    n_samples = 100
    z = torch.randn(n_samples, 100, device=device)
    
    with torch.no_grad():
        fake_images = G(z)
        d_scores = D(fake_images)
    
    # 分析判别器得分
    d_scores_np = d_scores.cpu().numpy()
    print("\n判别器得分统计:")
    print(f"平均值: {np.mean(d_scores_np):.4f}")
    print(f"标准差: {np.std(d_scores_np):.4f}")
    print(f"最小值: {np.min(d_scores_np):.4f}")
    print(f"最大值: {np.max(d_scores_np):.4f}")
    print(f"被判别为真实的比例: {np.mean(d_scores_np > 0.5):.2%}")
    
    # 分析生成图像的统计信息
    print("\n生成图像统计:")
    print(f"形状: {fake_images.shape}")
    print(f"最小值: {fake_images.min().item():.4f}")
    print(f"最大值: {fake_images.max().item():.4f}")
    print(f"均值: {fake_images.mean().item():.4f}")
    print(f"标准差: {fake_images.std().item():.4f}")
    
    # 保存一些示例图像
    show_imgs(fake_images[:16], "最终生成的样本", "improved_images/final_samples.png")

if __name__ == "__main__":
    # 设置随机种子以便复现
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    # 训练模型
    G, D = train_gan()
    
    # 分析结果
    analyze_results(G, D)