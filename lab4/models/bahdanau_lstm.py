import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        # Bahdanau cần output của encoder
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return outputs, hidden, cell

class BahdanauAttention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.W_a = nn.Linear(hid_dim * 2, hid_dim) # Concat encoder_out + decoder_hidden
        self.v_a = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden (decoder): [batch, hid_dim] (lấy lớp cuối)
        # encoder_outputs: [batch, src_len, hid_dim]
        src_len = encoder_outputs.shape[1]
        
        # Repeat decoder hidden state
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        # Calculate energy
        energy = torch.tanh(self.W_a(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v_a(energy).squeeze(2)
        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.attention = BahdanauAttention(hid_dim)
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim + hid_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        
        # Calculate attention weights
        # hidden[-1] is the last layer hidden state: [batch, hid_dim]
        a = self.attention(hidden[-1], encoder_outputs) 
        a = a.unsqueeze(1)
        
        # Weighted sum of encoder outputs
        weighted = torch.bmm(a, encoder_outputs) # [batch, 1, hid_dim]
        
        # Input to LSTM: embedding + context
        rnn_input = torch.cat((embedded, weighted), dim=2)
        
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        
        return prediction, hidden, cell

class Seq2SeqBahdanau(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        encoder_outputs, hidden, cell = self.encoder(src)
        input = trg[:, 0]
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[:, t, :] = output
            top1 = output.argmax(1) 
            input = trg[:, t] if random.random() < teacher_forcing_ratio else top1
            
        return outputs