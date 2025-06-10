import torch
import torch.nn as nn
from .base_model import BaseModel

class LSTM(BaseModel):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__(input_size, hidden_size, output_size)
        
        # 输入门组件
        self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)
        # 遗忘门组件
        self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size)
        # 输出门组件
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)
        # 单元状态组件
        self.cell_gate = nn.Linear(input_size + hidden_size, hidden_size)
        
        # 输出层
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, input, hidden_state):
        h, c = hidden_state
        
        # 合并输入和隐藏状态
        combined = torch.cat((input, h), 1)
        
        # 计算各个门的值
        i = torch.sigmoid(self.input_gate(combined))
        f = torch.sigmoid(self.forget_gate(combined))
        o = torch.sigmoid(self.output_gate(combined))
        g = torch.tanh(self.cell_gate(combined))
        
        # 更新单元状态和隐藏状态
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        
        # 计算输出
        output = self.output_layer(h_next)
        output = self.softmax(output)
        
        return output, (h_next, c_next)
    
    def initHidden(self):
        # 返回隐藏状态和单元状态
        return (torch.zeros(1, self.hidden_size),
                torch.zeros(1, self.hidden_size))
