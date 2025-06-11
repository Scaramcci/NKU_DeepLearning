import torch
import torch.nn as nn
from .base_model import BaseModel

class RNN(BaseModel):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__(input_size, hidden_size, output_size)
        
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        # 不在这里指定设备，而是在使用时移动到设备上
        # 这样可以保持模型的灵活性
        return torch.zeros(1, self.hidden_size)
