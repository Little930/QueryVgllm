import torch
import torch.nn as nn
import math

class PositionEmbeddingSine3D(nn.Module):
    """Sinusoidal positional embedding for 3D coordinates.
    Args:
        hidden_size: dimension of the output embedding.
        num_frequencies: number of frequency bands per axis (default 6).
        temperature: scaling factor for frequencies (default 10000).
    """
    def __init__(self, hidden_size: int, num_frequencies: int = 6, temperature: float = 10000.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_frequencies = num_frequencies
        self.temperature = temperature
        # total channels = 6 * num_frequencies (sin+cos for each axis)
        assert hidden_size % (6 * num_frequencies) == 0, "hidden_size must be divisible by 6 * num_frequencies"
        self.proj = nn.Linear(6 * num_frequencies, hidden_size)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Encode 3D coordinates.
        Args:
            coords: Tensor of shape (..., 3) containing (x, y, z).
        Returns:
            Tensor of shape (..., hidden_size).
        """
        # coords shape: (..., 3)
        # expand to (..., 3, 1)
        coords = coords.unsqueeze(-1)  # (..., 3, 1)
        freq_bands = torch.arange(self.num_frequencies, dtype=torch.float32, device=coords.device)
        freq = self.temperature ** (2 * (freq_bands // 2) / self.num_frequencies)
        # sin and cos for each axis
        sin = torch.sin(coords / freq)  # (..., 3, num_frequencies)
        cos = torch.cos(coords / freq)  # (..., 3, num_frequencies)
        # concatenate sin and cos per axis -> (..., 3, 2*num_frequencies)
        emb = torch.cat([sin, cos], dim=-1)  # (..., 3, 2*num_frequencies)
        # flatten last two dims -> (..., 6*num_frequencies)
        emb = emb.view(*emb.shape[:-2], -1)
        return self.proj(emb)

class PositionEmbeddingMLP(nn.Module):
    """Learnable MLP based positional embedding for 3D coordinates.
    Args:
        hidden_size: output dimension.
        mlp_hidden: hidden dimension of the MLP (default 128).
    """
    def __init__(self, hidden_size: int, mlp_hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Encode 3D coordinates.
        Args:
            coords: Tensor of shape (..., 3).
        Returns:
            Tensor of shape (..., hidden_size).
        """
        return self.mlp(coords)
