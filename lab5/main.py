import argparse
import torch
import torch.nn as nn
from models.classifier import TransformerClassifier
from models.tagger import TransformerTagger
from utils.logger import log_result
from utils.build_vocab import load_vocab
from utils.tokenizer import WordTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ex", choices=["1", "2"], required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.ex == "1":
        word2id, _ = load_vocab("vocab/vocab1.json")
        tokenizer = WordTokenizer(word2id, max_len=128)
        sample = tokenizer.encode("gói hàng cẩn thận chơi pubg mượt")
        print(sample)
        vocab_size = len(word2id)
        model = TransformerClassifier(vocab_size, num_labels=10)
        log_result("Running TASK 1: Domain Classification")

    else:
        word2id, _ = load_vocab("vocab/vocab2.json")
        tokenizer = WordTokenizer(word2id, max_len=128)
        sample = tokenizer.encode("gói hàng cẩn thận chơi pubg mượt")
        print(sample)
        vocab_size = len(word2id)
        model = TransformerTagger(vocab_size, num_tags=9)
        log_result("Running TASK 2: Sequence Labeling (NER)")

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print(f"Start training {args.task}...")
    

if __name__ == "__main__":
    main()
