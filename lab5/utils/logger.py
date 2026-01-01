import os
from datetime import datetime

def log_result(message, log_dir="logs",exercise="1"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'results{exercise}.txt')

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now()}] {message}\n")
    print(message,'\n')
