import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
class NER_Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.tag2idx = {}
        self.idx2tag = {}

    def __len__(self):
        return len(self.word2idx)

    def build(self, dataset):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.tag2idx  = {"O": 0}  
        
        for sample in dataset:
            for w in sample["words"]:
                if w not in self.word2idx:
                    self.word2idx[w] = len(self.word2idx)
            for t in sample["tags"]:
                if t not in self.tag2idx:
                    self.tag2idx[t] = len(self.tag2idx)

        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.idx2tag  = {v: k for k, v in self.tag2idx.items()}

    def encode_words(self, tokens):
        return [
            self.word2idx.get(w, self.word2idx["<UNK>"])
            for w in tokens
        ]

    def encode_tags(self, tags):
        return [
            self.tag2idx[t] for t in tags
        ]

class PhoNER_Dataset(Dataset):
    def __init__(self, df, vocab=None):
        self.df = df
        self.vocab = vocab

        if self.vocab is None:
            full_data = df.to_dict(orient="records")
            self.vocab = NER_Vocabulary()
            self.vocab.build(full_data)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        item = self.df.iloc[index]
        
        tokens = item["words"]
        tags   = item["tags"]

        word_ids = self.vocab.encode_words(tokens)
        tag_ids  = self.vocab.encode_tags(tags)

        return torch.tensor(word_ids), torch.tensor(tag_ids)

class NER_Collate:
    def __init__(self, pad_idx=0, pad_tag=-1):
        self.pad_idx = pad_idx
        self.pad_tag = pad_tag

    def __call__(self, batch):
        word_seqs = [item[0] for item in batch]
        tag_seqs  = [item[1] for item in batch]

        word_seqs = pad_sequence(word_seqs, batch_first=True, padding_value=self.pad_idx)
        tag_seqs  = pad_sequence(tag_seqs, batch_first=True, padding_value=self.pad_tag)
        
        lengths = torch.tensor([len(item[0]) for item in batch], dtype=torch.long)


        return word_seqs, tag_seqs, lengths
def get_phoner_loaders(args):
    print("Đang load PhoNER từ Local...")

    train_df = pd.read_json("data/train_word.json",lines=True)
    val_df   = pd.read_json("data/dev_word.json",lines=True)
    test_df  = pd.read_json("data/test_word.json",lines=True)

    print(f"Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # build vocab từ train
    train_ds = PhoNER_Dataset(train_df)
    vocab = train_ds.vocab

    # share vocab cho val, test
    val_ds = PhoNER_Dataset(val_df, vocab=vocab)
    test_ds = PhoNER_Dataset(test_df, vocab=vocab)

    pad_idx = vocab.word2idx["<PAD>"]

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=NER_Collate(pad_idx)
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=NER_Collate(pad_idx)
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=NER_Collate(pad_idx)
    )

    vocab_size = len(vocab.word2idx)
    tag_size   = len(vocab.tag2idx)

    return train_loader, val_loader, test_loader,vocab, vocab_size, tag_size
