import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size, drop_rate=0):
        super(DenseLayer, self).__init__()
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.drop_rate = drop_rate
        
    def forward(self, x):
        new_features = self.conv1(self.relu1(self.norm1(x)))
        new_features = self.conv2(self.relu2(self.norm2(new_features)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, bn_size, drop_rate=0):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate, bn_size, drop_rate))
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.conv(self.relu(self.norm(x)))
        return self.pool(x)


class DenseNet(nn.Module):
    def __init__(self, growth_rate=12, block_config=(6, 12, 24), num_init_features=24,
                 bn_size=4, compression_rate=0.5, drop_rate=0.2, num_classes=10):
        super(DenseNet, self).__init__()
        
        # 初始卷积层
        self.features = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True)
        )
        
        # DenseBlocks
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            # 添加DenseBlock
            block = DenseBlock(num_layers, num_features, growth_rate, bn_size, drop_rate)
            self.features.add_module(f'denseblock{i+1}', block)
            num_features = num_features + num_layers * growth_rate
            
            # 添加Transition层，除了最后一个block
            if i != len(block_config) - 1:
                transition = Transition(num_features, int(num_features * compression_rate))
                self.features.add_module(f'transition{i+1}', transition)
                num_features = int(num_features * compression_rate)
        
        # 最后的BN
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.features.add_module('relu5', nn.ReLU(inplace=True))
        self.features.add_module('avg_pool', nn.AdaptiveAvgPool2d((1, 1)))
        
        # 分类器
        self.classifier = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        features = self.features(x)
        out = features.view(features.size(0), -1)
        out = self.classifier(out)
        return out


def DenseNetCIFAR():
    return DenseNet(growth_rate=12, block_config=(6, 12, 24), num_init_features=24, 
                   compression_rate=0.5, drop_rate=0.2)
