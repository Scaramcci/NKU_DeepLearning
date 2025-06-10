import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        # 定义两层linear以及leaky relu激活函数
        self.fc1 = nn.Linear(z_dim, 256)
        self.fc2 = nn.Linear(256, 784)  # 28*28=784
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        # 实现生成器
        x = self.leaky_relu(self.fc1(x))
        out = self.fc2(x)
        
        out = torch.tanh(out) # range [-1, 1]
        # convert to image 
        out = out.view(out.size(0), 1, 28, 28)
        return out

class Discriminator(torch.nn.Module):
    def __init__(self, inp_dim=784):
        super(Discriminator, self).__init__()
        # 定义两层linear以及leaky relu激活函数
        self.fc1 = torch.nn.Linear(inp_dim, 128)
        self.fc2 = torch.nn.Linear(128, 1)
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x):
        x = x.view(x.size(0), 784) # flatten (bs x 1 x 28 x 28) -> (bs x 784)
        # 实现判别器
        x = self.leaky_relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x
