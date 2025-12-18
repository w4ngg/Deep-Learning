import torch
import torch.nn as nn
import time
from tqdm import tqdm

class Trainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_epoch(self, iterator, clip=1.0):
        self.model.train()
        epoch_loss = 0
        
        for src, trg in tqdm(iterator, desc="Training"):
            src = src.to(self.device)
            trg = trg.to(self.device)
            
            self.optimizer.zero_grad()
            
            # output: [batch, trg_len, output_dim]
            output = self.model(src, trg)
            
            output_dim = output.shape[-1]
            
            # trg = [batch, trg_len] -> loại bỏ token đầu <SOS> khi tính loss nếu muốn, 
            # nhưng thông thường output[:, 1:] so với trg[:, 1:]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            
            loss = self.criterion(output, trg)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
        return epoch_loss / len(iterator)