import torch.nn as nn
from .transformer import TransformerEncoder

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, num_labels):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size)
        self.classifier = nn.Linear(256, num_labels)

    def forward(self, input_ids, attention_mask):
        x = self.encoder(input_ids, attention_mask)
        cls_repr = x[:, 0]          # dùng token đầu
        logits = self.classifier(cls_repr)
        return logits
