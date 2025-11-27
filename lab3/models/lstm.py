import torch
import torch.nn as nn

class MyLSTM(nn.Module):
    def __init__(self,vocab_size, embed_dim=300, hidden_size=256, num_layers=5, num_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout = 0.2
        )
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        out, (h_n, c_n) = self.lstm(x)
        final = h_n[-1,:,:]
        logits = self.fc(final)
        return logits