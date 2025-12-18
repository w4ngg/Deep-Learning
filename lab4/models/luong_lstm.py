import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# Encoder giống Bahdanau (cần outputs)
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return outputs, hidden, cell

class LuongAttention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        # Phương pháp General: h_t^T * W * h_s
        self.W = nn.Linear(hid_dim, hid_dim)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: [batch, 1, hid_dim]
        # encoder_outputs: [batch, src_len, hid_dim]
        
        # Tính score: General
        # energy: [batch, src_len, hid_dim]
        energy = self.W(encoder_outputs) 
        
        # attention scores: [batch, 1, src_len]
        attention = torch.bmm(decoder_hidden, energy.permute(0, 2, 1))
        
        return F.softmax(attention, dim=2)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.attention = LuongAttention(hid_dim)
        self.embedding = nn.Embedding(output_dim, emb_dim)
        # Luong: RNN chạy trước, sau đó mới tính attention
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_concat = nn.Linear(hid_dim * 2, hid_dim)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        
        # 1. Run RNN step first
        rnn_output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        
        # 2. Calculate Attention
        # rnn_output: [batch, 1, hid_dim] (đây là h_t)
        a = self.attention(rnn_output, encoder_outputs) # [batch, 1, src_len]
        
        # 3. Calculate Context Vector
        context = torch.bmm(a, encoder_outputs) # [batch, 1, hid_dim]
        
        # 4. Concatenate and produce output
        concat_input = torch.cat((rnn_output, context), dim=2)
        concat_output = torch.tanh(self.fc_concat(concat_input))
        
        prediction = self.fc_out(concat_output.squeeze(1))
        
        return prediction, hidden, cell

class Seq2SeqLuong(nn.Module):
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