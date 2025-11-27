import torch
import torch.nn as nn

class MyGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_size=256, num_layers=5, num_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout = 0.2
        )
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        emb = self.embedding(x)
        _, h_n = self.gru(emb)
        out = h_n[-1]  
        logits = self.fc(out)
        return logits

