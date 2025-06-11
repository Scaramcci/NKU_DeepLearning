import torch
import numpy as np
from models import Generator
from utils import show_imgs
import matplotlib.pyplot as plt

def load_generator(model_path='models/generator_final.pth'):
    """加载训练好的生成器模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = Generator().to(device)
    G.load_state_dict(torch.load(model_path, map_location=device))
    G.eval()
    return G

def generate_images(G, z=None, n_images=8):
    """使用给定的随机数生成图像"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if z is None:
        z = torch.randn(n_images, 100).to(device)
    else:
        z = z.to(device)
    
    with torch.no_grad():
        images = G(z)
    show_imgs(images, save_path='images/generated.png')
    return images

def analyze_latent_space(G, n_samples=5, n_variations=3):
    """分析潜在空间"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 生成初始随机向量
    base_z = torch.randn(8, 100).to(device)
    print("基础随机向量生成完成")
    
    # 随机选择维度进行分析
    dims_to_analyze = np.random.choice(100, n_samples, replace=False)
    print(f"选择的维度: {dims_to_analyze}")
    
    # 对每个选择的维度进行变化
    for dim in dims_to_analyze:
        plt.figure(figsize=(15, 5))
        print(f"\n分析维度 {dim}:")
        
        # 生成不同的变化值
        variations = np.linspace(-2, 2, n_variations)
        
        for i, val in enumerate(variations):
            z = base_z.clone()
            z[:, dim] = val
            
            # 生成图像
            with torch.no_grad():
                images = G(z)
            
            plt.subplot(1, n_variations, i+1)
            show_imgs(images, title=f'Dim {dim}: {val:.2f}')
            
        plt.savefig(f'images/latent_dim_{dim}_analysis.png')
        plt.close()
        
def modify_dimension(G, z, dim, value):
    """修改指定维度的值并生成新图像"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    z = z.clone()
    z[:, dim] = value
    return generate_images(G, z)

if __name__ == "__main__":
    # 加载训练好的生成器
    G = load_generator()
    
    # 生成基础随机向量
    base_z = torch.randn(8, 100)
    
    # 生成基础图像
    print("生成基础图像:")
    generate_images(G, base_z)
    
    # 分析潜在空间
    print("\n开始潜在空间分析:")
    analyze_latent_space(G)
    
    # 示例：修改特定维度
    print("\n修改特定维度示例:")
    modify_dimension(G, base_z, dim=0, value=2.0)
