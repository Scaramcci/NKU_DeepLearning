import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        
    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        output, hidden = self.gru(embedded)
        return output, hidden

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 3, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / torch.sqrt(torch.FloatTensor([hidden_size]))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(1)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        
        energy = torch.tanh(self.attn(torch.cat((h, encoder_outputs), 2)))
        energy = energy.transpose(2, 1)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        attention = torch.bmm(v, energy)
        
        return F.softmax(attention.squeeze(1), dim=1)

class AttentionDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=50):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = self.attention(hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        context = context.transpose(0, 1)

        rnn_input = torch.cat((embedded, context), 2)
        output, hidden = self.gru(rnn_input, hidden)

        output = output.squeeze(0)
        context = context.squeeze(0)
        output = F.log_softmax(self.out(output), dim=1)
        
        return output, hidden, attn_weights

class AttentionSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(AttentionSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, input_seq, target_seq, teacher_forcing_ratio=0.5):
        batch_size = input_seq.size(1)
        target_length = target_seq.size(0)
        target_vocab_size = self.decoder.output_size

        outputs = torch.zeros(target_length, batch_size, target_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(input_seq)

        decoder_input = torch.tensor([[0]], device=self.device)  # SOS token
        decoder_hidden = hidden

        use_teacher_forcing = True if torch.rand(1).item() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            for t in range(1, target_length):
                decoder_output, decoder_hidden, _ = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                outputs[t] = decoder_output
                decoder_input = target_seq[t]
        else:
            for t in range(1, target_length):
                decoder_output, decoder_hidden, _ = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                outputs[t] = decoder_output
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()

        return outputs
