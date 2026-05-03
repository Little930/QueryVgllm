"""3D Position Encoding modules for world coordinate embeddings.

Migrated from 3DRS project. Provides sinusoidal and MLP-based 3D position
encodings that map world coordinates (x, y, z) into embedding space.
"""

import torch
import torch.nn as nn


class PositionEmbeddingSine3D(nn.Module):
    """Sinusoidal 3D position embedding.
    
    Encodes 3D world coordinates (x, y, z) using sinusoidal functions,
    similar to the positional encoding in "Attention Is All You Need",
    generalized to 3D space.
    
    Args:
        embedding_size: Output embedding dimension.
        temperature: Temperature for frequency scaling.
        n_points: Number of sampled points per patch (1, 5, or 9).
    """

    def __init__(self, embedding_size, temperature=10000, n_points=1):
        super(PositionEmbeddingSine3D, self).__init__()
        self.embedding_size = embedding_size
        self.temperature = temperature
        self.n_points = n_points

    def forward(self, x):
        """
        Args:
            x: World coordinates tensor.
               Shape: [B, N, 3] when n_points=1
               Shape: [B, N, n_points, 3] when n_points > 1
        Returns:
            Position embeddings of shape [B, N, embedding_size]
        """
        num_feats = self.embedding_size // (3 * self.n_points)

        if self.n_points > 1:
            x = x.flatten(1, 2)
        B, N, _ = x.shape

        dim_t = torch.arange(num_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / num_feats)

        pos_x = x[:, :, 0][..., None] / dim_t
        pos_y = x[:, :, 1][..., None] / dim_t
        pos_z = x[:, :, 2][..., None] / dim_t
        if num_feats % 2 != 0:
            pos_x = torch.cat([pos_x, torch.zeros(B, N, 1).to(pos_x.device)], dim=-1)
            pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)[..., :-1]
            pos_y = torch.cat([pos_y, torch.zeros(B, N, 1).to(pos_y.device)], dim=-1)
            pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)[..., :-1]
            pos_z = torch.cat([pos_z, torch.zeros(B, N, 1).to(pos_z.device)], dim=-1)
            pos_z = torch.stack((pos_z[:, :, 0::2].sin(), pos_z[:, :, 1::2].cos()), dim=3).flatten(2)[..., :-1]
        else:
            pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
            pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
            pos_z = torch.stack((pos_z[:, :, 0::2].sin(), pos_z[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat([pos_x, pos_y, pos_z], dim=2)
        if self.n_points > 1:
            pos = pos.view(B, N // self.n_points, self.n_points * 3 * num_feats)

        out = torch.zeros((B, N // self.n_points, self.embedding_size), dtype=x.dtype, device=x.device)
        out[:, :, :pos.shape[2]] = pos

        return out


class PositionEmbeddingMLP(nn.Module):
    """MLP-based 3D position embedding.
    
    Uses a learnable MLP to project 3D world coordinates into embedding space.
    
    Args:
        embedding_size: Output embedding dimension.
        temperature: Unused, kept for API compatibility with PositionEmbeddingSine3D.
        n_points: Number of sampled points per patch (1, 5, or 9).
    """

    def __init__(self, embedding_size, temperature=10000, n_points=1):
        super(PositionEmbeddingMLP, self).__init__()
        self.embedding_size = embedding_size
        self.temperature = temperature
        self.n_points = n_points
        self.hidden_size = 512
        self.mlp = nn.Sequential(
            nn.Linear(3, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.embedding_size)
        )

    def forward(self, x):
        """
        Args:
            x: World coordinates tensor.
               Shape: [B, N, 3] when n_points=1
               Shape: [B, N, n_points, 3] when n_points > 1
        Returns:
            Position embeddings of shape [B, N, embedding_size]
        """
        if self.n_points > 1:
            x = x.flatten(1, 2)
        B, N, _ = x.shape

        pos = self.mlp(x)  # (B, N, embedding_size)

        if self.n_points > 1:
            # When n_points > 1, reshape back and pad to full embedding_size
            num_feats = self.embedding_size // (3 * self.n_points)
            pos = pos.view(B, N // self.n_points, self.n_points * self.embedding_size)
            out = torch.zeros((B, N // self.n_points, self.embedding_size), dtype=x.dtype, device=x.device)
            out[:, :, :pos.shape[2]] = pos[:, :, :self.embedding_size]
            return out

        return pos
