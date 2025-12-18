# evaluate.py (chỉ cập nhật phần decode)
import torch
from rouge_score import rouge_scorer
from tqdm import tqdm

class Evaluator:
    def __init__(self, model, device, src_sp, trg_sp):
        self.model = model
        self.device = device
        self.src_sp = src_sp
        self.trg_sp = trg_sp
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)

    def decode_prediction(self, indices):
        # indices: List[int] hoặc Tensor
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
            
        # Loại bỏ các token đặc biệt trước khi decode
        cleaned_indices = []
        for idx in indices:
            if idx == self.trg_sp.eos_id():
                break
            if idx not in [self.trg_sp.bos_id(), self.trg_sp.pad_id()]:
                cleaned_indices.append(idx)
        
        # SentencePiece decode: ids -> string
        return self.trg_sp.DecodeIds(cleaned_indices)
    def translate_sentence(self, sentence, max_len=50):
        """
        Hàm thực hiện Greedy Decoding cho mô hình LSTM.
        Input: Câu tiếng Anh (str)
        Output: Câu tiếng Việt (str)
        """
        self.model.eval()
        
        # 1. Tokenize & Encode
        # Thêm BOS và EOS
        src_ids = [self.src_sp.bos_id()] + self.src_sp.EncodeAsIds(sentence) + [self.src_sp.eos_id()]
        src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(self.device) # [1, src_len]
        
        # 2. Encoder Phase
        with torch.no_grad():
            # Xử lý sự khác biệt giữa Bài 1 (Basic) và Bài 2/3 (Attention)
            # Bài 1 encoder trả về: hidden, cell
            # Bài 2/3 encoder trả về: outputs, hidden, cell
            encoder_result = self.model.encoder(src_tensor)
            
            if len(encoder_result) == 3:
                encoder_outputs, hidden, cell = encoder_result
                has_attention = True
            else:
                hidden, cell = encoder_result
                encoder_outputs = None
                has_attention = False

        # 3. Decoder Phase (Greedy Loop)
        # Bắt đầu với token <SOS>
        trg_indexes = [self.trg_sp.bos_id()]
        
        # Lặp tối đa max_len lần để sinh từ
        for i in range(max_len):
            # Lấy token cuối cùng vừa sinh ra để làm input cho bước tiếp theo
            trg_tensor = torch.tensor([trg_indexes[-1]], dtype=torch.long).to(self.device) # [1] (vì decoder code cũ handle unsqueeze)

            with torch.no_grad():
                if has_attention:
                    # Decoder có Attention cần encoder_outputs
                    output, hidden, cell = self.model.decoder(trg_tensor, hidden, cell, encoder_outputs)
                else:
                    # Decoder cơ bản
                    output, hidden, cell = self.model.decoder(trg_tensor, hidden, cell)

                # output: [1, output_dim] -> Chọn từ có xác suất cao nhất (Greedy)
                pred_token = output.argmax(1).item()
                
                trg_indexes.append(pred_token)

                # Nếu gặp thẻ <EOS> thì dừng
                if pred_token == self.trg_sp.eos_id():
                    break
        
        # 4. Decode về text
        translated_text = self.decode_prediction(trg_indexes)
        return translated_text

    def inference_sentences(self, sentences):
        """
        Nhận vào một list các câu tiếng Anh và in ra bản dịch.
        """
        print("\n" + "="*50)
        print("KẾT QUẢ DỊCH THỬ (INFERENCE):")
        print("="*50)
        
        for text in sentences:
            translated = self.translate_sentence(text)
            print(f"Input : {text}")
            print(f"Output: {translated}")
            print("-" * 50)
    def evaluate(self, iterator):
        self.model.eval()
        epoch_loss = 0
        all_refs = []
        all_hyps = []
        criterion = torch.nn.CrossEntropyLoss(ignore_index=self.trg_sp.pad_id())

        with torch.no_grad():
            for src, trg in tqdm(iterator, desc="Evaluating"):
                src = src.to(self.device)
                trg = trg.to(self.device)

                output = self.model(src, trg, teacher_forcing_ratio=0) 
                
                output_dim = output.shape[-1]
                output_loss = output[:, 1:].reshape(-1, output_dim)
                trg_loss = trg[:, 1:].reshape(-1)
                
                loss = criterion(output_loss, trg_loss)
                epoch_loss += loss.item()

                pred_token_indices = output.argmax(2)
                
                for i in range(src.shape[0]):
                    # Target gốc
                    ref_indices = trg[i].tolist()
                    # Decode target để lấy string chuẩn
                    ref = self.decode_prediction(ref_indices)
                    
                    # Decode prediction
                    hyp = self.decode_prediction(pred_token_indices[i])
                    
                    all_refs.append(ref)
                    all_hyps.append(hyp)

        total_rouge_l = 0
        for ref, hyp in zip(all_refs, all_hyps):
            scores = self.scorer.score(ref, hyp)
            total_rouge_l += scores['rougeL'].fmeasure
        
        avg_rouge = total_rouge_l / len(all_refs) if all_refs else 0
        avg_loss = epoch_loss / len(iterator)

        return avg_loss, avg_rouge