import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from cnn_models import Generator, Discriminator
from utils import show_imgs

def analyze_gan():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    G = Generator().to(device)
    D = Discriminator().to(device)
    
    try:
        G.load_state_dict(torch.load('models/generator_final.pth', map_location=device))
        D.load_state_dict(torch.load('models/discriminator_final.pth', map_location=device))
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    G.eval()
    D.eval()
    
    # 生成图像并评估
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
    
    # 检查模式崩溃问题
    fake_images_flat = fake_images.view(n_samples, -1).cpu().numpy()
    distances = []
    for i in range(min(10, n_samples)):
        for j in range(i+1, min(10, n_samples)):
            dist = np.linalg.norm(fake_images_flat[i] - fake_images_flat[j])
            distances.append(dist)
    
    print("\n生成图像多样性分析:")
    print(f"样本间平均距离: {np.mean(distances):.4f}")
    print(f"样本间距离标准差: {np.std(distances):.4f}")
    print(f"样本间最小距离: {np.min(distances):.4f}")
    print(f"样本间最大距离: {np.max(distances):.4f}")
    
    # 生成一些示例图像
    sample_z = torch.randn(16, 100, device=device)
    with torch.no_grad():
        sample_images = G(sample_z)
    
    # 保存示例图像
    show_imgs(sample_images, "生成的样本图像", "images/analysis_samples.png")
    print("\n示例图像已保存到 images/analysis_samples.png")

if __name__ == "__main__":
    analyze_gan()