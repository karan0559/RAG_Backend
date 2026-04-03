from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

class Reranker:
    def __init__(self, model_name=None, device=None):
        model_name = model_name or os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
  
        self.model.to(self.device)
        self.model.eval()

    def rerank(self, query: str, passages: list[str], top_n=5, batch_size: int = 16):
        if not passages:
            return [], []

        all_scores = []
        for i in range(0, len(passages), batch_size):
            batch = passages[i:i + batch_size]
            pairs = [(query, passage) for passage in batch]
            inputs = self.tokenizer.batch_encode_plus(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                scores = self.model(**inputs).logits.squeeze(-1)

            if scores.dim() == 0:
                all_scores.append(scores)
            else:
                all_scores.extend(scores)

        scores_tensor = torch.stack(all_scores) if isinstance(all_scores[0], torch.Tensor) else torch.tensor(all_scores)

        sorted_indices = torch.argsort(scores_tensor, descending=True)
        reranked = [passages[i] for i in sorted_indices[:top_n]]
        return reranked, scores_tensor[sorted_indices[:top_n]].tolist()
