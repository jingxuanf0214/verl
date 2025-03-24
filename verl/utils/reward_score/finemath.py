import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Global variables for the model and tokenizer
_tokenizer = None
_model = None
_device = None

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def _get_embedding(text):
    global _tokenizer, _model, _device
    
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3")
        _model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model.to(_device)
        _model.eval()
    
    # Use the model's encode method which handles pooling internally
    with torch.no_grad():
        task = "text-matching"
        embedding = _model.encode([text], task=task, device=_device)[0]
        #embedding = torch.tensor(embedding, device=_device)
        #embedding = embedding / embedding.norm(dim=1, keepdim=True)  # Normalize embedding
    
    return embedding # np array

def compute_score(solution_str, ground_truth):
    """Compute reward score based on embedding similarity between solution and ground truth.
    
    Args:
        solution_str: The model's generated solution
        ground_truth: The ground truth answer
    
    Returns:
        float: A score between 0 and 1 based on embedding similarity
    """
    
    solution_embedding = _get_embedding(solution_str)
    ground_truth_embedding = _get_embedding(ground_truth)
    
    # Compute cosine similarity
    similarity = cosine_similarity(solution_embedding, ground_truth_embedding)
    
    # Convert to float and ensure it's between 0 and 1
    return float((similarity + 1) / 2)  # Scale from [-1, 1] to [0, 1]
   