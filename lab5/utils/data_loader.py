import torch
from torch.utils.data import Dataset, DataLoader


class ClassificationDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(
                self.encodings["input_ids"][idx], dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                self.encodings["attention_mask"][idx], dtype=torch.long
            ),
            "labels": torch.tensor(
                self.labels[idx], dtype=torch.long
            ),
        }


class NERDataset(Dataset):
    def __init__(self, encodings, tag_ids):
        self.encodings = encodings
        self.tag_ids = tag_ids

    def __len__(self):
        return len(self.tag_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(
                self.encodings["input_ids"][idx], dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                self.encodings["attention_mask"][idx], dtype=torch.long
            ),
            "labels": torch.tensor(
                self.tag_ids[idx], dtype=torch.long
            ),
        }


def build_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
