import torch
import torch.nn as nn
import torch.optim as optim
import argparse

from utils.dataset import get_loaders
from utils.ner_dataset import get_phoner_loaders
from models.lstm import MyLSTM
from models.gru import MyGRU
from models.bi_lstm import MyBiLSTM
from trainer import train_epoch, evaluate, train_epoch_ner, evaluate_ner
from sklearn.metrics import classification_report
from seqeval.metrics import classification_report as seqeval_classification_report


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10,help='nums of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for Adam')
    parser.add_argument('--exercise', type=int,help='Choose which exercise to run',default=1)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Thiết bị huấn luyện (cuda/cpu)')

    args = parser.parse_args()
    return args

def main():
    # 1. Lấy tham số từ dòng lệnh
    args = get_args()
    
    print("="*30)
    print("CẤU HÌNH CHẠY:")
    for arg in vars(args):
        print(f"--{arg}: {getattr(args, arg)}")
    print("="*30)

    device = torch.device(args.device)
    if args.exercise in [1,2]:
        train_loader, val_loader, test_loader, vocab_size = get_loaders(args)
    elif args.exercise == 3:
        train_loader, val_loader, test_loader, vocab, vocab_size, tag_size = get_phoner_loaders(args)

    print(f"Kích thước từ điển (Vocab Size): {vocab_size}")
    print(f"Số lượng mẫu Train: {len(train_loader)}")

    if args.exercise == 1:
        model = MyLSTM(vocab_size=vocab_size).to(device)
    elif args.exercise == 2:
        model = MyGRU(vocab_size=vocab_size).to(device)
    elif args.exercise ==3:
        model = MyBiLSTM(vocab_size=vocab_size,embedding_dim=300,
                         hidden_dim=256,num_layers=5,output_dim=tag_size,pad_idx=0).to(device)
    
    if args.exercise in [1,2]:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        best_val_f1 = 0.0
        
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            
            train_loss, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_f1, _, _ = evaluate(model, val_loader, criterion, device)
            
            print(f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
            print(f"Val Loss:   {val_loss:.4f} | Val F1:   {val_f1:.4f}")
            save_path = 'logs/'+'exercise_'+str(args.exercise)+'_best_model.pth'
            # Lưu model tốt nhất
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(),save_path )
                print(f"--> Đã lưu model tốt nhất tại: {save_path}")

        # 6. Testing
        print("\n================ ĐÁNH GIÁ TRÊN TẬP TEST ================")
        model.load_state_dict(torch.load(save_path))
        test_loss, test_f1, true_labels, pred_labels = evaluate(model, test_loader, criterion, device)
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test F1 (Weighted): {test_f1:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(true_labels, pred_labels, target_names=['Tích cực', 'Tiêu cực', 'Trung tính']))
    elif args.exercise == 3:
        idx2tag = vocab.idx2tag
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        best_val_f1 = 0.0

        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch + 1}/{args.epochs}")

            train_loss, train_f1 = train_epoch_ner(model, train_loader, criterion, optimizer, device,idx2tag)
            val_loss, val_f1, _, _ = evaluate_ner(model, val_loader, criterion, device,idx2tag)

            print(f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
            print(f"Val Loss:   {val_loss:.4f} | Val F1:   {val_f1:.4f}")

            save_path = 'logs/'+'exercise_'+str(args.exercise)+'_best_model.pth'

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), save_path)
                print(f"--> Đã lưu model tốt nhất: {save_path}")

        # --- TEST ---
        print("\n================ ĐÁNH GIÁ TRÊN TẬP TEST (NER) ================")

        model.load_state_dict(torch.load(save_path))

        test_loss, test_f1, true_tags, pred_tags = evaluate_ner(model, test_loader, criterion, device,idx2tag)

        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test F1:   {test_f1:.4f}")

        print("\nSequence Labeling Report:")
        print(seqeval_classification_report(true_tags, pred_tags))

if __name__ == "__main__":
    main()