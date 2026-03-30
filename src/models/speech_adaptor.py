"""Speech Adapter for aligning speech embeddings to text embedding space."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeechAdapter(nn.Module):
	"""Project temporal speech representations into text embedding space."""

	def __init__(
		self,
		input_dim: int = 1024,
		output_dim: int = 1024,
		downsample_factor: int = 4,
	):
		super().__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.downsample_factor = downsample_factor

		self.downsample = nn.AvgPool1d(
			kernel_size=downsample_factor,
			stride=downsample_factor,
			padding=0,
		)
		self.projection = nn.Linear(input_dim, output_dim)
		self.layer_norm = nn.LayerNorm(output_dim)

	def forward(self, speech_representations: torch.Tensor) -> torch.Tensor:
		# Expect (batch, seq_len, hidden)
		x = speech_representations.transpose(1, 2)
		if x.size(-1) >= self.downsample_factor:
			x = self.downsample(x)
		else:
			# Keep forward stable for very short sequences.
			x = x.mean(dim=-1, keepdim=True)
		x = x.transpose(1, 2)
		x = x.mean(dim=1)
		x = self.projection(x)
		x = self.layer_norm(x)
		return F.normalize(x, p=2, dim=1)

	def get_embedding_dim(self) -> int:
		return self.output_dim

