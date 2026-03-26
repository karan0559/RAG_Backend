"""
Singleton model loader — prevents loading the same model multiple times.
All services import from here to share a single instance in memory.
"""
from sentence_transformers import SentenceTransformer

_models: dict = {}


def get_model(model_name: str) -> SentenceTransformer:
    """Return cached model instance, loading it only once."""
    if model_name not in _models:
        print(f"[ModelLoader] Loading model: {model_name}")
        _models[model_name] = SentenceTransformer(model_name)
        print(f"[ModelLoader] Model ready: {model_name}")
    return _models[model_name]
