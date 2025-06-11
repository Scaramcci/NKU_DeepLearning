import torch
import torch.nn as nn
from .base_model import BaseModel


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU, self).__init__()
        
        self.hidden_size = hidden_size
        
        # 重置门组件
        self.reset_gate = nn.Linear(input_size + hidden_size, hidden_size)
        # 更新门组件
        self.update_gate = nn.Linear(input_size + hidden_size, hidden_size)
        # 候选隐藏状态组件
        self.h_tilde = nn.Linear(input_size + hidden_size, hidden_size)
        
        # 输出层
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, input, hidden):
        # 合并输入和隐藏状态
        combined = torch.cat((input, hidden), 1)
        
        # 计算门的值
        r = torch.sigmoid(self.reset_gate(combined))
        z = torch.sigmoid(self.update_gate(combined))
        
        # 计算候选隐藏状态
        combined_reset = torch.cat((input, r * hidden), 1)
        h_tilde = torch.tanh(self.h_tilde(combined_reset))
        
        # 更新隐藏状态
        h_next = (1 - z) * hidden + z * h_tilde
        
        # 计算输出
        output = self.output_layer(h_next)
        output = self.softmax(output)
        
        return output, h_next
    
    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
