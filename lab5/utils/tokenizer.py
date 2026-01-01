from utils.build_vocab import normalize_text
class WordTokenizer:
    def __init__(self, word2id, max_len=128):
        self.word2id = word2id
        self.max_len = max_len

        self.pad_id = word2id["<PAD>"]
        self.unk_id = word2id["<UNK>"]
        self.cls_id = word2id["<CLS>"]
        self.sep_id = word2id["<SEP>"]

    def encode(self, text: str):
        text = normalize_text(text)
        words = text.split()

        ids = [self.cls_id]

        for w in words:
            ids.append(self.word2id.get(w, self.unk_id))

        ids.append(self.sep_id)

        ids = ids[: self.max_len]

        attention_mask = [1] * len(ids)

        # padding
        while len(ids) < self.max_len:
            ids.append(self.pad_id)
            attention_mask.append(0)

        return {
            "input_ids": ids,
            "attention_mask": attention_mask
        }
