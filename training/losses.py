"""Loss functions for distillation training"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


class DistillationLoss(nn.Module):
    """
    Distillation loss for aligning speech embeddings to text embeddings.
    
    Compares audio embedding (from adapter) with text embedding (ground truth).
    """
    
    def __init__(
        self,
        loss_type: Literal["mse", "cosine", "both"] = "cosine",
        alpha: float = 0.5,
        normalize_for_cosine: bool = True
    ):
        """
        Args:
            loss_type: Type of loss - "mse", "cosine", or "both"
            alpha: Weight for cosine loss when loss_type="both"
            normalize_for_cosine: Whether to L2-normalize embeddings before cosine
        """
        super().__init__()
        self.loss_type = loss_type
        self.alpha = alpha
        self.normalize_for_cosine = normalize_for_cosine

        # Loss functions
        self.mse_loss = nn.MSELoss()
    
    def forward(
        self,
        audio_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute distillation loss.
        
        Args:
            audio_embeddings: Embeddings from speech adapter
                Shape: (batch_size, embedding_dim)
            text_embeddings: Target embeddings from text encoder
                Shape: (batch_size, embedding_dim)
        
        Returns:
            Loss scalar
        """

        # print(f"Audio for cosine shape: {audio_embeddings.shape}, Text for cosine shape: {text_embeddings.shape}")
        assert audio_embeddings.shape == text_embeddings.shape, "Audio and text embeddings must have the same shape"

        if self.loss_type == "mse":
            # Mean Squared Error loss
            loss = self.mse_loss(audio_embeddings, text_embeddings)
        
        elif self.loss_type == "cosine":
            # Cosine similarity loss (1 - cosine_similarity)

            if self.normalize_for_cosine:
                audio_for_cosine = F.normalize(audio_embeddings, p=2, dim=-1)
                text_for_cosine = F.normalize(text_embeddings, p=2, dim=-1)
            else:
                audio_for_cosine = audio_embeddings
                text_for_cosine = text_embeddings
                
            # print(f"Audio for cosine shape: {audio_for_cosine.shape}, Text for cosine shape: {text_for_cosine.shape}")

            cosine_sim = F.cosine_similarity(audio_for_cosine, text_for_cosine, dim=-1)
            loss = (1 - cosine_sim).mean()
        
        elif self.loss_type == "both":
            # Combined MSE and cosine loss
            if self.normalize_for_cosine:
                audio_for_cosine = F.normalize(audio_embeddings, p=2, dim=-1)
                text_for_cosine = F.normalize(text_embeddings, p=2, dim=-1)
            else:
                audio_for_cosine = audio_embeddings
                text_for_cosine = text_embeddings

            mse = self.mse_loss(audio_embeddings, text_embeddings)
            cosine_sim = F.cosine_similarity(audio_for_cosine, text_for_cosine, dim=-1)
            cosine = (1 - cosine_sim).mean()
            loss = (1 - self.alpha) * mse + self.alpha * cosine
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss
    
    def compute_similarity(
        self,
        audio_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine similarity between embeddings (for evaluation).
        
        Args:
            audio_embeddings: Embeddings from speech adapter
            text_embeddings: Embeddings from text encoder
        
        Returns:
            Cosine similarity scores (batch_size,)
        """
        if self.normalize_for_cosine:
            audio_embeddings = F.normalize(audio_embeddings, p=2, dim=-1)
            text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        return F.cosine_similarity(audio_embeddings, text_embeddings, dim=-1)

