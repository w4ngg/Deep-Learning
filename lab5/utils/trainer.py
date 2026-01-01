import torch

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    task: str
):
    """
    task: "1" or "2"
    """
    model.train()
    total_loss = 0

    for batch in dataloader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, mask)

        if task == "1":
            loss = criterion(outputs, labels)

        elif task == "2":
            loss = criterion(
                outputs.view(-1, outputs.size(-1)),
                labels.view(-1)
            )
        else:
            raise ValueError("Unknown task")

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
