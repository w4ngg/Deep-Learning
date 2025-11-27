import torch
from sklearn.metrics import f1_score
import numpy as np
from tqdm import tqdm
from seqeval.metrics import f1_score as seqeval_f1
from seqeval.metrics import classification_report as seqeval_classification_report
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    for texts, labels in progress_bar:
        texts, labels = texts.to(device), labels.to(device)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # Lưu kết quả để tính F1
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
    avg_loss = epoch_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, f1

def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for texts, labels in dataloader:
            texts, labels = texts.to(device), labels.to(device)
            
            outputs = model(texts)
            loss = criterion(outputs, labels)
            
            epoch_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = epoch_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, f1, all_labels, all_preds



def train_epoch_ner(model, dataloader, criterion, optimizer, device,idx2tag):
    model.train()
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    total_loss = 0
    true_all, pred_all = [], []

    for tokens, labels,_ in progress_bar:
        tokens = tokens.to(device)
        labels = labels.to(device)
        print('shape của tokens:',tokens.shape)
        print('shape của labels:',labels.shape)
        logits = model(tokens)  # [B, L, C]
        print('shape của logits:',logits.shape)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # predict
        preds = torch.argmax(logits, dim=-1)  # [B, L]

        for p, t in zip(preds, labels):
            pred_sent = []
            true_sent = []           
            for pred_idx, true_idx in zip(p, t):
                if true_idx.item() != -1:   # bỏ qua vị trí pad
                    pred_sent.append(idx2tag[pred_idx.item()])
                    true_sent.append(idx2tag[true_idx.item()])
            pred_all.append(pred_sent)
            true_all.append(true_sent)

    f1 = seqeval_f1(true_all, pred_all)
    return total_loss / len(dataloader), f1
def evaluate_ner(model, dataloader, criterion, device,idx2tag):
    model.eval()

    total_loss = 0
    true_all, pred_all = [], []

    with torch.no_grad():
        for tokens, labels,_ in dataloader:
            tokens = tokens.to(device)
            labels = labels.to(device)

            logits = model(tokens) #return [Batch,Seq_len of a sentence, nums of tags ]
            print('shape của logits:',logits.shape)
            ##flatten the logits and labels
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            for p, t in zip(preds, labels):
                pred_sent = []
                true_sent = []           
                for pred_idx, true_idx in zip(p, t):
                    if true_idx.item() != -1:   # bỏ qua vị trí pad
                        pred_sent.append(idx2tag[pred_idx.item()])
                        true_sent.append(idx2tag[true_idx.item()])
                pred_all.append(pred_sent)
                true_all.append(true_sent)
                
    f1 = seqeval_f1(true_all, pred_all)
    return total_loss / len(dataloader), f1, true_all, pred_all

