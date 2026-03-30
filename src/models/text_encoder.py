"""Text Encoder supporting E5-Mistral-7B-Instruct, Qwen3-Embedding-0.6B, and Qwen3-Embedding-4B"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Union, Optional


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Last token pooling for Qwen3-Embedding models (recommended for Qwen3-Embedding-4B).
    
    Args:
        last_hidden_states: Hidden states from model output, shape (batch_size, seq_len, hidden_dim)
        attention_mask: Attention mask, shape (batch_size, seq_len)
    
    Returns:
        Pooled embeddings, shape (batch_size, hidden_dim)
    """
    # Handle left/right padding per sample to avoid batch-level assumptions.
    sequence_lengths = attention_mask.sum(dim=1) - 1
    sequence_lengths = sequence_lengths.clamp_min(0)
    last_positions = attention_mask.shape[1] - 1
    last_indices = torch.where(
        attention_mask[:, -1].bool(),
        torch.full_like(sequence_lengths, last_positions),
        sequence_lengths,
    )
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        last_indices,
    ]


class TextEncoder(nn.Module):
    """
    Text encoder wrapper supporting multiple embedding models:
    - E5-Mistral-7B-Instruct (4096 dim)
    - Qwen3-Embedding-0.6B (1024 dim)
    - Qwen3-Embedding-4B (2560 dim)
    
    Generates normalized text embeddings.
    For Qwen3-Embedding-4B, uses last token pooling (recommended by the model authors).
    For other models, uses mean pooling or pooler_output.
    """
    
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen3-Embedding-0.6B", 
        freeze: bool = True,
        instruction: Optional[str] = None
    ):
        """
        Args:
            model_name: HuggingFace model identifier
            freeze: Whether to freeze model parameters
            instruction: Custom instruction for Qwen3 models (optional)
        """
        super().__init__()
        self.model_name = model_name
        self.instruction = instruction
        
        # Detect model type
        model_name_lower = model_name.lower()
        self.is_qwen3 = "qwen3" in model_name_lower
        self.is_e5 = "e5" in model_name.lower()
        
        # Detect if it's Qwen3-Embedding-4B (uses last token pooling)
        self.is_qwen3_4b = "qwen3-embedding-4b" in model_name_lower
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Configure tokenizer padding side for Qwen3-Embedding-4B (recommended for better performance)
        if self.is_qwen3_4b:
            self.tokenizer.padding_side = "left"
        
        self.model = AutoModel.from_pretrained(model_name)
        
        # Freeze all parameters
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
        
        # Get embedding dimension
        # Qwen3-Embedding-0.6B has configurable dimensions (32-1024), default is 1024
        # Qwen3-Embedding-4B has configurable dimensions (32-2560), default is 2560
        # Check if model has embedding_size attribute (Qwen3) or hidden_size (E5)
        if hasattr(self.model.config, 'embedding_size'):
            self.embedding_dim = self.model.config.embedding_size
        elif hasattr(self.model.config, 'hidden_size'):
            self.embedding_dim = self.model.config.hidden_size
        else:
            # Fallback: try to infer from model output
            with torch.no_grad():
                dummy_input = self.tokenizer("test", return_tensors="pt")
                dummy_output = self.model(**dummy_input)
                if hasattr(dummy_output, 'last_hidden_state'):
                    self.embedding_dim = dummy_output.last_hidden_state.shape[-1]
                elif hasattr(dummy_output, 'pooler_output') and dummy_output.pooler_output is not None:
                    self.embedding_dim = dummy_output.pooler_output.shape[-1]
                else:
                    raise ValueError(f"Could not determine embedding dimension for {model_name}")
        
        # Verify if it's 4B model by dimension (2560) if not detected by name
        if self.is_qwen3 and self.embedding_dim == 2560 and not self.is_qwen3_4b:
            self.is_qwen3_4b = True
    
    def encode(
        self,
        texts: Union[str, List[str]],
        max_length: int = 512,
        normalize: bool = True,
        device: str = None
    ) -> torch.Tensor:
        """
        Encode text(s) into embeddings.
        
        Args:
            texts: Single text string or list of texts
            max_length: Maximum sequence length
            normalize: Whether to L2-normalize embeddings
            device: Device to run on (auto-detect if None)
        
        Returns:
            Embeddings tensor of shape (batch_size, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if device is None:
            device = next(self.model.parameters()).device
        
        # Format texts according to model type
        if self.is_e5:
            # E5 models expect input format: "query: {text}" or "passage: {text}"
            formatted_texts = [f"query: {text}" for text in texts]
        elif self.is_qwen3:
            # Qwen3 supports custom instructions
            # If instruction is provided, use it; otherwise use default format
            if self.instruction:
                formatted_texts = [f"{self.instruction}\n{text}" for text in texts]
            else:
                # Default: just use the text as-is (Qwen3 can work without prefix)
                formatted_texts = texts
        else:
            # Unknown model type, use texts as-is
            formatted_texts = texts
        
        encoded = self.tokenizer(
            formatted_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.model(**encoded)
            
            # Extract embeddings based on model type
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                # Some models have pooler_output (e.g., BERT-style)
                embeddings = outputs.pooler_output
            elif hasattr(outputs, 'last_hidden_state'):
                attention_mask = encoded["attention_mask"]
                
                # Qwen3-Embedding-4B uses last token pooling (recommended)
                if self.is_qwen3_4b:
                    embeddings = last_token_pool(outputs.last_hidden_state, attention_mask)
                else:
                    # Use mean pooling over sequence length for other models
                    # Sum embeddings over sequence dimension
                    mask = attention_mask.unsqueeze(-1).to(outputs.last_hidden_state.dtype)
                    embeddings = outputs.last_hidden_state * mask
                    # Average over sequence length (excluding padding)
                    lengths = attention_mask.sum(dim=1, keepdim=True).clamp_min(1)
                    embeddings = embeddings.sum(dim=1) / lengths
            else:
                # Fallback: try to get embeddings from outputs
                if hasattr(outputs, 'embeddings'):
                    embeddings = outputs.embeddings
                    if embeddings.ndim != 2:
                        raise ValueError(
                            f"Fallback embeddings must be 2D [batch, dim], got shape {tuple(embeddings.shape)}"
                        )
                else:
                    raise ValueError(f"Could not extract embeddings from model {self.model_name}")
        
        # L2 normalize
        if normalize:
            embeddings = nn.functional.normalize(embeddings, p=2, dim=-1)
        
        return embeddings
    
    def forward(self, texts: Union[str, List[str]], **kwargs) -> torch.Tensor:
        """Forward pass for compatibility with nn.Module"""
        return self.encode(texts, **kwargs)

