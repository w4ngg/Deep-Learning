import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

import os
from pyvi import ViTokenizer 

class Vocabulary:
    '''build vocab using pyvi tokenizer'''
    def __init__(self):
        self.vocab = []
        self.word2idx = {}
        self.idx2word = {}
    def __len__(self):
        return len(self.word2idx)
    def build_vocabulary(self, sentence_list):
        for sentence in sentence_list:
            tokenized_sentence = ViTokenizer.tokenize(sentence)
            self.vocab = self.vocab + tokenized_sentence.split()
        self.vocab = list(set(self.vocab))
        self.word2idx = {w: (idx+2) for idx,w in enumerate(self.vocab)}
        self.word2idx['<PAD>'] = 0
        self.word2idx['<UNK>'] = 1
        self.idx2word = {idx : w for w, idx in self.word2idx.items()}
    def encoding(self,sentence):
        tokenized_sentence = ViTokenizer.tokenize(sentence)
        tokenized_sentence = tokenized_sentence.split()
        return [
            self.word2idx.get(token, self.word2idx['<UNK>'])
            for token in tokenized_sentence
        ]
        

class VSFC_Dataset(Dataset):
    def __init__(self, dataframe, vocab=None):
        
        self.df = dataframe
        self.vocab = vocab
        self.text_col = 'sentence'
        self.label_col = 'sentiment'
        if self.vocab is None:
            self.vocab = Vocabulary()
            self.vocab.build_vocabulary(self.df[self.text_col].tolist())
        label2idx = {"negative": 0,
                     "neutral": 1,
                     "positive": 2}
        self.df[self.label_col] = self.df[self.label_col].map(label2idx)
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        text = row[self.text_col]
        label = int(row[self.label_col]) 
        assert self.vocab is not None, "Vocab chưa được khởi tạo!" 
        encoded_text = self.vocab.encoding(text)
        
        return torch.tensor(encoded_text), torch.tensor(label)

class Collate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        # Lọc bỏ các mẫu bị lỗi (nếu có)
        batch = [item for item in batch if item[0].size(0) > 0]
        
        texts = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        
        if len(texts) == 0:
            return torch.tensor([]), torch.tensor([])

        texts = pad_sequence(texts, batch_first=True, padding_value=self.pad_idx)
        labels = torch.tensor(labels)
        
        return texts, labels

def get_loaders(args):
    def load_file(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Không tìm thấy file tại: {path}")
        if path.endswith('.csv'):
            return pd.read_csv(path)
        elif path.endswith('.json'):
            return pd.read_json(path)
        elif path.endswith('.xlsx'):
            return pd.read_excel(path)
        elif path.endswith('.tsv') or path.endswith('.txt'):
            return pd.read_csv(path, sep='\t')
        else:
            raise ValueError("Định dạng file không hỗ trợ. Hãy dùng csv, xlsx hoặc tsv.")

    print("Đang tải dữ liệu từ Local...")
    
    if args.exercise == 1 or args.exercise == 2:
        train_df = load_file('data\\UIT-VSFC-train.json')
        val_df = load_file('data\\UIT-VSFC-dev.json')
        test_df = load_file('data\\UIT-VSFC-test.json')
    else:
        train_df = load_file()
        val_df = load_file()
        test_df = load_file()
    print(f"Load xong: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    train_ds = VSFC_Dataset(train_df) 
    val_ds = VSFC_Dataset(val_df, vocab=train_ds.vocab) 
    test_ds = VSFC_Dataset(test_df, vocab=train_ds.vocab)
    vocab_size = len(train_ds.vocab)
    pad_idx = train_ds.vocab.word2idx["<PAD>"]
    
    # 3. Tạo DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=Collate(pad_idx)
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=Collate(pad_idx)
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=Collate(pad_idx)
    )
    
    return train_loader, val_loader, test_loader, vocab_size