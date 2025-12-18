# utils/dataset.py
import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class NMTDataset(Dataset):
    def __init__(self, json_file, src_sp, trg_sp):
        self.src_sp = src_sp
        self.trg_sp = trg_sp
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        src_text = item['english']
        trg_text = item['vietnamese']

        # SentencePiece encode: text -> list of ids
        # Chúng ta tự thêm BOS và EOS để kiểm soát tốt hơn
        src_indices = [self.src_sp.bos_id()] + \
                      self.src_sp.EncodeAsIds(src_text) + \
                      [self.src_sp.eos_id()]
        
        trg_indices = [self.trg_sp.bos_id()] + \
                      self.trg_sp.EncodeAsIds(trg_text) + \
                      [self.trg_sp.eos_id()]

        return torch.tensor(src_indices), torch.tensor(trg_indices)

class Collate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        src_batch, trg_batch = [], []
        for src_item, trg_item in batch:
            src_batch.append(src_item)
            trg_batch.append(trg_item)
        
        # Padding
        src_batch = pad_sequence(src_batch, padding_value=self.pad_idx, batch_first=True)
        trg_batch = pad_sequence(trg_batch, padding_value=self.pad_idx, batch_first=True)
        
        return src_batch, trg_batch