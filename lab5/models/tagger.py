import torch.nn as nn
from .transformer import TransformerEncoder

class TransformerTagger(nn.Module):
    def __init__(self, vocab_size, num_tags):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size)
        self.fc = nn.Linear(256, num_tags)

    def forward(self, input_ids, attention_mask):
        x = self.encoder(input_ids, attention_mask)
        logits = self.fc(x)
        return logits
