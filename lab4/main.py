import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
from utils.build_vocab import train_sentencepiece, load_spm_processor
from utils.dataset import NMTDataset, Collate
from utils.logger import log_results
from utils.trainer import Trainer
from utils.evaluate import Evaluator
from models.lstm import Encoder as Enc1, Decoder as Dec1, Seq2Seq as Seq1
from models.bahdanau_lstm import Encoder as Enc2, Decoder as Dec2, Seq2SeqBahdanau as Seq2
from models.luong_lstm import Encoder as Enc3, Decoder as Dec3, Seq2SeqLuong as Seq3

def get_model(exercise_idx, input_dim, output_dim, device):
    enc_emb = 256
    dec_emb = 256
    hid_dim = 256
    n_layers = 3
    dropout = 0.5
    if exercise_idx == 1:
        enc = Enc1(input_dim, enc_emb, hid_dim, n_layers, dropout)
        dec = Dec1(output_dim, dec_emb, hid_dim, n_layers, dropout)
        model = Seq1(enc, dec, device)
    elif exercise_idx == 2:
        enc = Enc2(input_dim, enc_emb, hid_dim, n_layers, dropout)
        dec = Dec2(output_dim, dec_emb, hid_dim, n_layers, dropout)
        model = Seq2(enc, dec, device)
    elif exercise_idx == 3:
        enc = Enc3(input_dim, enc_emb, hid_dim, n_layers, dropout)
        dec = Dec3(output_dim, dec_emb, hid_dim, n_layers, dropout)
        model = Seq3(enc, dec, device)
    else:
        raise ValueError("Exercise must be 1, 2, or 3")
    return model.to(device)

def main():
    parser = argparse.ArgumentParser(description='NMT PhoMT Training')
    parser.add_argument('--ex', type=int, required=True, help='Exercise number: 1, 2, or 3')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    args = parser.parse_args()
    # ConfigsDATA_DIR = 'data'
    DATA_DIR ='data'
    TRAIN_FILE = os.path.join(DATA_DIR, 'small-train.json')
    DEV_FILE = os.path.join(DATA_DIR, 'small-dev.json')
    TEST_FILE = os.path.join(DATA_DIR, 'small-test.json')
    LOG_FILE = f'logs/logs_ex{args.ex}.txt'
    # Config cho SentencePiece
    SPM_PREFIX = 'vocab/spm_phomt'
    VOCAB_SIZE = 14440 
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Exercise {args.ex} on {DEVICE}")
    # 1. Train & Load SentencePiece Model
    if not os.path.exists(f'{SPM_PREFIX}.model'):
        print("Training SentencePiece models (Unigram)...")
        train_sentencepiece(TRAIN_FILE, SPM_PREFIX, VOCAB_SIZE, model_type='unigram')
    print("Loading SentencePiece models...")
    sp_model = load_spm_processor(f'{SPM_PREFIX}.model')
    src_sp = sp_model
    trg_sp = sp_model
    # 2. Dataset & Dataloader
    train_dataset = NMTDataset(TRAIN_FILE, src_sp, trg_sp)
    dev_dataset = NMTDataset(DEV_FILE, src_sp, trg_sp)
    test_dataset = NMTDataset(TEST_FILE, src_sp, trg_sp)

    collate_fn = Collate(pad_idx=src_sp.pad_id())
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    # 3. Init Model
    model = get_model(args.ex, src_sp.get_piece_size(), trg_sp.get_piece_size(), DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    PAD_IDX = trg_sp.pad_id()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    trainer = Trainer(model, optimizer, criterion, DEVICE)
    evaluator = Evaluator(model, DEVICE, src_sp, trg_sp)

    # 4. Training Loop
    best_valid_loss = float('inf')

    for epoch in range(args.epochs):
        train_loss = trainer.train_epoch(train_loader)
        valid_loss, valid_rouge = evaluator.evaluate(dev_loader)
        
        log_results(LOG_FILE, epoch+1, train_loss, valid_rouge)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'logs/model_ex{args.ex}.pt')
    
    # 5. Final Test Evaluation
    print("Loading best model for testing...")
    model.load_state_dict(torch.load(f'logs/model_ex{args.ex}.pt', map_location=DEVICE))
    test_loss, test_rouge = evaluator.evaluate(test_loader)
    log_results(LOG_FILE, "TEST", test_loss, 0, test_rouge)
    print(f"Final Test ROUGE-L: {test_rouge:.4f}")
    sample_sentences = [
        "It begins with a countdown .",
        "I want to go to school.",
        "The weather is very beautiful today.",
        "Machine learning is fascinating."
    ]
    
    translated = evaluator.inference_sentences(sample_sentences)
    print(f"Input: {sample_sentences}")
    print(f"Output: {translated}")
if __name__ == '__main__':
    main()