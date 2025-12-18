import sentencepiece as spm
import json
import os

def train_sentencepiece(json_file, model_prefix, vocab_size, model_type='unigram'):
    """
    Huấn luyện mô hình SentencePiece sử dụng Shared Vocab).
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # combine for shared vocab
    combined_text = [item['english'] for item in data] + [item['vietnamese'] for item in data]
    
    joint_tmp_file = f'{model_prefix}_joint.txt'
    
    with open(joint_tmp_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(combined_text))
        
    # 2. params
    cmd_args = (
        f'--input={joint_tmp_file} '
        f'--model_prefix={model_prefix} '
        f'--vocab_size={vocab_size} '
        f'--model_type={model_type} '
        f'--pad_id=0 --bos_id=1 --eos_id=2 --unk_id=3 '
        f'--character_coverage=1.0'
    )
    
    # 3. train sp
    spm.SentencePieceTrainer.train(cmd_args)
    
    print(f"Done training Shared SentencePiece model with prefix: {model_prefix}")

    # Xóa file tạm sau khi train xong
    if os.path.exists(joint_tmp_file):
        os.remove(joint_tmp_file)

def load_spm_processor(model_path):
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp