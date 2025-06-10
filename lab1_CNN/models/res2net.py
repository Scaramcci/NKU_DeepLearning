import torch
import torch.nn as nn
import torch.nn.functional as F


class Res2NetBottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, scales=4, base_width=26, scale_factor=4):
        super(Res2NetBottleneck, self).__init__()
        width = int(planes * (base_width / 64.0))
        self.conv1 = nn.Conv2d(inplanes, width*scales, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width*scales)
        
        self.scales = scales
        self.width = width
        self.scale_factor = scale_factor
        
        convs = []
        bns = []
        for i in range(scales):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride if i == 0 else 1, 
                                  padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        
        self.conv3 = nn.Conv2d(width*scales, planes*self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        residual = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        
        spx = torch.split(out, self.width, 1)
        spo = []
        sp = spx[0]
        
        sp = self.relu(self.bns[0](self.convs[0](sp)))
        spo.append(sp)
        
        for i in range(1, self.scales):
            if i < len(spx):
                sp = spx[i]
                sp = sp + spo[i-1] if self.scale_factor == 1 else spo[i-1]
                sp = self.relu(self.bns[i](self.convs[i](sp)))
                spo.append(sp)
        
        out = torch.cat(spo, 1)
        out = self.bn3(self.conv3(out))
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out


class Res2Net(nn.Module):
    def __init__(self, block, layers, num_classes=10, scales=4, base_width=26):
        super(Res2Net, self).__init__()
        self.inplanes = 64
        self.scales = scales
        self.base_width = base_width
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.scales, self.base_width))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, scales=self.scales, base_width=self.base_width))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


def Res2NetCIFAR(num_classes=10, scales=4, base_width=26):
    return Res2Net(Res2NetBottleneck, [2, 2, 2, 2], num_classes=num_classes,
                  scales=scales, base_width=base_width)
