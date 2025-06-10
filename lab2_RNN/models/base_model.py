import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BaseModel, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, input, hidden):
        raise NotImplementedError

    def initHidden(self):
        raise NotImplementedError
