import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=100, ngf=64):
        """
        改进的DCGAN生成器
        z_dim: 输入噪声维度
        ngf: 生成器特征图数量
        """
        super(Generator, self).__init__()
        self.z_dim = z_dim
        
        # 输入是一个z_dim维的噪声向量，输出是(1, 28, 28)的图像
        # 确保输出大小正好是28x28
        
        self.main = nn.Sequential(
            # 输入: z_dim x 1 x 1
            # 第一层：转置卷积，将1x1扩展到4x4
            nn.ConvTranspose2d(z_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 状态大小: (ngf*8) x 4 x 4
            
            # 第二层：4x4 -> 7x7
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 状态大小: (ngf*4) x 7 x 7
            
            # 第三层：7x7 -> 14x14
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 状态大小: (ngf*2) x 14 x 14
            
            # 第四层：14x14 -> 28x28 (精确匹配FashionMNIST大小)
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 状态大小: ngf x 28 x 28
            
            # 最后一层：保持28x28大小不变，只改变通道数
            nn.Conv2d(ngf, 1, 3, 1, 1, bias=False),
            nn.Tanh()
            # 输出状态大小: 1 x 28 x 28
        )
        
        # 权重初始化
        self.apply(self._weights_init)
    
    def _weights_init(self, m):
        """权重初始化函数"""
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    def forward(self, input):
        # 将输入reshape为适合卷积的形状
        if input.dim() == 2:
            input = input.view(input.size(0), input.size(1), 1, 1)
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, ndf=64):
        """
        改进的DCGAN判别器
        ndf: 判别器特征图数量
        """
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            # 输入: 1 x 28 x 28
            # 第一层：28x28 -> 14x14
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),  # 增加dropout比例
            # 状态大小: ndf x 14 x 14
            
            # 第二层：14x14 -> 7x7
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),  # 增加dropout比例
            # 状态大小: (ndf*2) x 7 x 7
            
            # 第三层：7x7 -> 4x4
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),  # 增加dropout比例
            # 状态大小: (ndf*4) x 4 x 4
            
            # 第四层：4x4 -> 1x1
            # 使用3x3卷积核和padding=0，确保能处理3x3的输入
            nn.Conv2d(ndf * 4, 1, 3, 1, 0, bias=False),
            nn.Sigmoid()
            # 输出状态大小: 1 x 1 x 1
        )
        
        # 权重初始化
        self.apply(self._weights_init)
    
    def _weights_init(self, m):
        """权重初始化函数"""
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    def forward(self, input):
        output = self.main(input)
        # 确保输出形状与标签匹配
        # 首先打平输出，然后确保长度与批次大小匹配
        batch_size = input.size(0)
        # 计算每个样本的平均值，确保输出形状为 [batch_size]
        return output.view(batch_size, -1).mean(dim=1)

# 测试模型
if __name__ == "__main__":
    # 测试生成器
    z_dim = 100
    batch_size = 64
    
    G = Generator(z_dim)
    D = Discriminator()
    
    # 生成随机噪声
    noise = torch.randn(batch_size, z_dim)
    
    # 生成假图像
    fake_images = G(noise)
    print(f"生成器输出形状: {fake_images.shape}")
    
    # 判别器判断
    output = D(fake_images)
    print(f"判别器输出形状: {output.shape}")
    
    # 打印模型参数数量
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"生成器参数数量: {count_parameters(G):,}")
    print(f"判别器参数数量: {count_parameters(D):,}")