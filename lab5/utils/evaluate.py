import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from utils.logger import log_result

def evaluate_classification(model, dataloader, device):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, mask)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    return {
        "accuracy": acc,
        "f1": f1
    }
def evaluate_ner(model, dataloader, device, id2tag):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, mask)
            preds = torch.argmax(logits, dim=-1)

            preds = preds.cpu().numpy()
            labels = labels.cpu().numpy()

            for p_seq, l_seq in zip(preds, labels):
                for p, l in zip(p_seq, l_seq):
                    if l == -100:
                        continue
                    all_preds.append(p)
                    all_labels.append(l)

    acc = accuracy_score(all_labels, all_preds)

    # loại tag "O" khi tính F1
    valid_idx = [
        i for i, l in enumerate(all_labels)
        if id2tag[l] != "O"
    ]

    if len(valid_idx) == 0:
        f1 = 0.0
    else:
        f1 = f1_score(
            np.array(all_labels)[valid_idx],
            np.array(all_preds)[valid_idx],
            average="micro"
        )

    return {
        "accuracy": acc,
        "f1": f1
    }
def evaluate(
    model,
    dataloader,
    device,
    task: str,
    id2tag=None,
    split_name="dev"
):
    """
    Evaluate model on dev or test set

    Args:
        model: trained model
        dataloader: dev/test dataloader  
        id2tag: required for NER
        split_name: "dev" or "test"

    Returns:
        metrics dict
    """

    if task == "1":
        metrics = evaluate_classification(
            model,
            dataloader,
            device
        )

        message = (
            f"[{split_name.upper()}] "
            f"Acc: {metrics['accuracy']:.4f} | "
            f"F1: {metrics['f1']:.4f}"
        )
        log_result(message, exercise="1")
        print(message)

    elif task == "2":
        if id2tag is None:
            raise ValueError("id2tag is required for NER evaluation")

        metrics = evaluate_ner(
            model,
            dataloader,
            device,
            id2tag
        )

        message = (
            f"[{split_name.upper()}] "
            f"Token Acc: {metrics['accuracy']:.4f} | "
            f"NER F1: {metrics['f1']:.4f} \n"
        )
        log_result(message, exercise="2")
        print(message)

    else:
        raise ValueError("Unknown task type")

    return metrics