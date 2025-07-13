from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class Reranker:
    def __init__(self, model_name="BAAI/bge-reranker-large", device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model.to(self.device)

    def rerank(self, query: str, passages: list[str], top_n=5):
        pairs = [(query, passage) for passage in passages]
        inputs = self.tokenizer.batch_encode_plus(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            scores = self.model(**inputs).logits.squeeze(-1)

        sorted_indices = torch.argsort(scores, descending=True)
        reranked = [passages[i] for i in sorted_indices[:top_n]]
        return reranked, scores[sorted_indices[:top_n]].tolist()
