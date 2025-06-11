import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np

def show_imgs(x, title=None, save_path=None):
    """显示图像网格"""
    plt.figure(figsize=(10, 10))
    grid = vutils.make_grid(x.detach().cpu(), nrow=8, normalize=True, pad_value=0.3)
    grid = grid.transpose(0,2).transpose(0,1)
    plt.imshow(grid.numpy())
    
    if title:
        plt.title(title)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def calculate_diversity(imgs, n_pairs=1000):
    """
    计算生成图像的多样性
    
    参数:
        imgs (Tensor): 图像张量，形状为 (N, C, H, W)
        n_pairs (int): 用于计算的图像对数量
    
    返回:
        float: 平均距离
        float: 距离标准差
    """
    if not isinstance(imgs, torch.Tensor):
        imgs = torch.Tensor(imgs)
    
    # 将图像展平为向量
    imgs_flat = imgs.view(imgs.size(0), -1)
    
    # 随机选择图像对
    n = imgs.size(0)
    n_pairs = min(n_pairs, n * (n - 1) // 2)  # 确保不超过可能的对数
    
    distances = []
    for _ in range(n_pairs):
        i, j = np.random.choice(n, 2, replace=False)
        # 计算欧氏距离
        dist = torch.norm(imgs_flat[i] - imgs_flat[j]).item()
        distances.append(dist)
    
    # 计算平均距离和标准差
    avg_dist = np.mean(distances)
    std_dist = np.std(distances)
    
    return avg_dist, std_dist

def interpolate_latent(G, z1, z2, n_steps=10, device='cpu'):
    """
    在潜在空间中进行线性插值
    
    参数:
        G: 生成器模型
        z1 (Tensor): 起始潜在向量
        z2 (Tensor): 结束潜在向量
        n_steps (int): 插值步数
        device (str): 设备
    
    返回:
        Tensor: 插值生成的图像
    """
    # 确保输入是张量并且在正确的设备上
    if not isinstance(z1, torch.Tensor):
        z1 = torch.Tensor(z1).to(device)
    if not isinstance(z2, torch.Tensor):
        z2 = torch.Tensor(z2).to(device)
    
    # 生成插值系数
    alphas = np.linspace(0, 1, n_steps)
    
    # 在潜在空间中进行线性插值
    z_interp = []
    for alpha in alphas:
        z = (1 - alpha) * z1 + alpha * z2
        z_interp.append(z)
    
    # 将插值向量堆叠成一个批次
    z_interp = torch.stack(z_interp)
    
    # 使用生成器生成图像
    with torch.no_grad():
        imgs = G(z_interp)
    
    return imgs
