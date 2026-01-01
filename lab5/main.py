import argparse
import torch
import torch.nn as nn
from models.classifier import TransformerClassifier
from models.tagger import TransformerTagger
from utils.logger import log_result
from utils.build_vocab import load_vocab
from utils.tokenizer import WordTokenizer
from utils.load_data import load_phoner_dataset, load_viodc_dataset, load_viodc_dataset_with_mapping, load_phoner_dataset_json_with_mapping
from utils.data_loader import build_dataloader, NERDataset,ClassificationDataset
from utils.trainer import train_one_epoch
from utils.evaluate import evaluate
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
        print(f'Testing: \n {sample}')
        vocab_size = len(word2id)

        encodings, labels, label2id, id2label = load_viodc_dataset("data/train.json",tokenizer)
        dataset = ClassificationDataset(encodings, labels)
        dataloader = build_dataloader(dataset, batch_size=16)

        enc_dev, y_dev = load_viodc_dataset_with_mapping( "data/dev.json",tokenizer,label2id        )
        enc_test, y_test = load_viodc_dataset_with_mapping("data/test.json",tokenizer,label2id)

        dev_dataset = ClassificationDataset(enc_dev, y_dev)
        test_dataset = ClassificationDataset(enc_test, y_test)

        dev_loader = build_dataloader(dev_dataset, batch_size=16, shuffle=False)
        test_loader = build_dataloader(test_dataset,  batch_size=16, shuffle=False)
        model = TransformerClassifier(vocab_size, num_labels=4)
        log_result("\n Running TASK 1: Domain Classification", exercise=args.ex)
        print(label2id)

    else:
        word2id, _ = load_vocab("vocab/vocab2.json")
        tokenizer = WordTokenizer(word2id, max_len=128)
        sample = tokenizer.encode("gói hàng cẩn thận chơi pubg mượt")
        print(sample)
        vocab_size = len(word2id)
        encodings, tag_ids, tag2id,id2tag = load_phoner_dataset("data/train_syllable.json",tokenizer)
        dataset = NERDataset(encodings, tag_ids)
        dataloader = build_dataloader(dataset,batch_size=16)
        model = TransformerTagger(vocab_size, num_tags=21)

        enc_dev, y_dev = load_phoner_dataset_json_with_mapping( "data/dev_syllable.json",tokenizer,tag2id        )
        enc_test, y_test = load_phoner_dataset_json_with_mapping("data/test_syllable.json",tokenizer,tag2id)

        dev_dataset = NERDataset(enc_dev, y_dev)
        test_dataset = NERDataset(enc_test, y_test)
        dev_loader = build_dataloader(dev_dataset, batch_size=16, shuffle=False)
        test_loader = build_dataloader(test_dataset,  batch_size=16, shuffle=False)
        log_result("\n Running TASK 2: Sequence Labeling (NER)", exercise=args.ex)
        print(tag2id)


    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print(f"Start training exercise {args.ex}...")
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, dataloader, optimizer, criterion, device, args.ex)
        log_result(f"Epoch {epoch+1} | Train Loss: {loss:.4f}", exercise=args.ex)

    # ---- DEV EVALUATION ----
        evaluate(
            model,
            dev_loader,
            device,
            task=args.ex,
            id2tag=id2tag if args.ex == "2" else None,
            split_name="dev"
        )
    # ---- TEST EVALUATION ----
    evaluate(
        model,
        test_loader,
        device,
        task=args.ex,
        id2tag=id2tag if args.ex == "2" else None,
        split_name="test"
    )
if __name__ == "__main__":
    main()
