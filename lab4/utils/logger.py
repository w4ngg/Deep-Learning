import os
import datetime

def log_results(log_file, epoch, train_loss, dev_score, test_score=None):
    mode = 'a' if os.path.exists(log_file) else 'w'
    with open(log_file, mode, encoding='utf-8') as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] Epoch: {epoch} | Train Loss: {train_loss:.4f} | Dev ROUGE-L: {dev_score:.4f}"
        if test_score is not None:
            log_line += f" | Test ROUGE-L: {test_score:.4f}"
        f.write(log_line + "\n")
        print(log_line)