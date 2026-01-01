"""Word-level vocabulary
Run this file first to build vocab files"""

import json
import unicodedata
from collections import Counter

SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<CLS>", "<SEP>"]

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return text.lower().strip()
def normalize_word(word):
    return unicodedata.normalize("NFC", word.lower())

def build_vocab_from_json(
    json_path: str,
    min_freq: int = 2,
    max_vocab_size: int = 32000
):
    """
    Build word-level vocab from UIT-ViOCD train.json
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    counter = Counter()

    for sample in data.values():
        text = normalize_text(sample["review"])
        words = text.split()
        counter.update(words)

    # lọc theo min_freq
    vocab_words = [
        word for word, freq in counter.items()
        if freq >= min_freq
    ]

    # giới hạn vocab size
    if max_vocab_size:
        vocab_words = sorted(
            vocab_words,
            key=lambda w: counter[w],
            reverse=True
        )[: max_vocab_size - len(SPECIAL_TOKENS)]

    # build vocab
    word2id = {}
    idx = 0

    for token in SPECIAL_TOKENS:
        word2id[token] = idx
        idx += 1

    for word in vocab_words:
        word2id[word] = idx
        idx += 1

    id2word = {v: k for k, v in word2id.items()}

    return word2id, id2word
def build_word_vocab_for_ner(
    json_path,
    min_freq=1,
    max_vocab_size=None
):
    counter = Counter()

    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            for w in item["words"]:
                counter.update([normalize_word(w)])

    vocab_words = [
        w for w, freq in counter.items()
        if freq >= min_freq
    ]

    if max_vocab_size:
        vocab_words = sorted(
            vocab_words,
            key=lambda w: counter[w],
            reverse=True
        )[: max_vocab_size - len(SPECIAL_TOKENS)]

    word2id = {}
    idx = 0

    for t in SPECIAL_TOKENS:
        word2id[t] = idx
        idx += 1

    for w in vocab_words:
        word2id[w] = idx
        idx += 1

    id2word = {v: k for k, v in word2id.items()}
    return word2id, id2word   
def save_vocab(word2id, path="vocab.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(word2id, f, ensure_ascii=False, indent=2)


def load_vocab(path="vocab.json"):
    with open(path, "r", encoding="utf-8") as f:
        word2id = json.load(f)
    id2word = {int(v): k for k, v in word2id.items()}
    return word2id, id2word

if __name__ == "__main__":
    word2id, id2word = build_vocab_from_json(
    "data/train.json",
    min_freq=2,
    max_vocab_size=32000
    )
    save_vocab(word2id, "vocab/vocab1.json")
    word2id_ner, id2word_ner = build_word_vocab_for_ner(
        "data/train_syllable.json",
        min_freq=1,
        max_vocab_size=32000
    )
    save_vocab(word2id_ner, "vocab/vocab2.json")