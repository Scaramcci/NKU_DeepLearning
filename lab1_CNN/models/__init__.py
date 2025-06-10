from .cnn import Net
from .resnet import ResNet18, ResNet34
from .densenet import DenseNetCIFAR
from .se_resnet import SEResNet18, SEResNet34
from .res2net import Res2NetCIFAR

__all__ = [
    'Net',  # Original CNN
    'ResNet18', 'ResNet34',  # ResNet
    'DenseNetCIFAR',  # DenseNet
    'SEResNet18', 'SEResNet34',  # SE-ResNet
    'Res2NetCIFAR',  # Res2Net
]
