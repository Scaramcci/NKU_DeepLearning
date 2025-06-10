import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self, device):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, input_seq, target_seq, teacher_forcing_ratio=0.5):
        input_length = input_seq.size(0)
        target_length = target_seq.size(0)
        target_vocab_size = self.decoder.out.out_features
        
        outputs = torch.zeros(target_length, target_vocab_size, device=self.device)
        
        encoder_hidden = self.encoder.init_hidden(self.device)
        _, encoder_hidden = self.encoder(input_seq[0], encoder_hidden)
        
        decoder_input = torch.tensor([[0]], device=self.device)  # SOS token
        decoder_hidden = encoder_hidden
        
        use_teacher_forcing = True if torch.rand(1).item() < teacher_forcing_ratio else False
        
        if use_teacher_forcing:
            for t in range(1, target_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                outputs[t] = decoder_output
                decoder_input = target_seq[t].unsqueeze(0)
        else:
            for t in range(1, target_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                outputs[t] = decoder_output
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach().unsqueeze(0)
        
        return outputs
