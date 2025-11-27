import torch.nn as nn
import torch

class MyBiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, output_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            bidirectional=True,
                            batch_first=True,
                            dropout=0.5 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len]
        embedded = self.embedding(x) # [batch_size, seq_len, emb_dim]
        
        # LSTM output
        outputs, (hidden, cell) = self.lstm(embedded)
        # outputs shape: [batch_size, seq_len, hidden_dim * 2]
        
        predictions = self.fc(outputs) # [batch_size, seq_len, output_dim]
        
        return predictions