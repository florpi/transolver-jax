import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Callable, Any


class PhysicsAttention(nn.Module):
    num_slices: int
    num_channels: int
    num_heads: int
    dim_per_head: int

    @nn.compact
    def __call__(self, x):
        batch_size, num_points, channels = x.shape

        # Dense layer
        x = nn.Dense(self.num_channels)(x)  # [B, N, C]
        # Project input to get slice weights (similar to Eq. 1 in paper)
        slice_logits = nn.Dense(self.num_slices)(x)  # [B, N, M]
        slice_weights = jax.nn.softmax(slice_logits, axis=-1)  # [B, N, M]

        slices = slice_weights[...,None] *  x[...,None, :] # [B, N, M, C]
        # Step 2: Aggregate slices to tokens (z_j in the paper) using Eq. (2)
        # z_j = sum(s_j) / sum(w_i,j)
        token_numerator = jnp.sum(slices, axis=1) 
        token_denominator = jnp.sum(slice_weights, axis=1,) 
        # z = token_numerator / token_denominator
        tokens = token_numerator / (token_denominator[...,None] + 1e-9)  # [B, M, C]
        qkv = nn.Dense(3 * self.num_heads * self.dim_per_head)(tokens)  # [B, M, 3 * H * D]
        qkv = qkv.reshape(
            batch_size, self.num_slices, 3, self.num_heads, self.dim_per_head
        )
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))  # [3, B, H, M, D]
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, H, M, D]


        scale = jnp.sqrt(self.dim_per_head)
        attention = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) / scale
        attention = jax.nn.softmax(attention, axis=-1)  # [B, H, M, M]

        token_output = jnp.matmul(attention, v)  # [B, H, M, D]
        token_output = jnp.transpose(token_output, (0, 2, 1, 3))
        slice_weights = jnp.transpose(slice_weights, (0, 2, 1))
        output = jnp.sum(token_output[:,:,None,...] * slice_weights[...,None, None], axis=1)  # [B, N, H, D]

        output = output.reshape(
            batch_size, num_points, self.num_heads * self.dim_per_head
        )
        return nn.Dense(channels)(output)


class TransolverBlock(nn.Module):
    num_slices: int
    num_heads: int
    dim_per_head: int
    mlp_dim: int
    num_channels: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, train: bool = True):
        # Layer normalization
        y = nn.LayerNorm()(x)

        # Physics attention
        y = PhysicsAttention(
            num_slices=self.num_slices,
            num_heads=self.num_heads,
            dim_per_head=self.dim_per_head,
            num_channels=self.num_channels,
        )(y)

        # Residual connection
        x = x + y

        # MLP block
        y = nn.LayerNorm()(x)
        y = nn.Dense(self.mlp_dim)(y)
        y = nn.gelu(y)
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not train)
        y = nn.Dense(x.shape[-1])(y)
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not train)

        # Residual connection
        return x + y

class Transolver(nn.Module):
    """Transolver model for point clouds."""
    num_layers: int
    num_slices: int
    num_heads: int
    hidden_dim: int
    num_channels: int
    mlp_dim: int
    output_dim: int
    dropout_rate: float = 0.0
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        
        # Initial embedding
           
        x = nn.Dense(self.hidden_dim)(x)
        
        # Transolver blocks
        for _ in range(self.num_layers):
            x = TransolverBlock(
                num_slices=self.num_slices,
                num_heads=self.num_heads,
                dim_per_head=self.hidden_dim // self.num_heads,
                mlp_dim=self.mlp_dim,
                num_channels=self.num_channels,
                dropout_rate=self.dropout_rate
            )(x, train=train)
        
        # Output projection
        return nn.Dense(self.output_dim)(x)

        
# class TransolverSummarizer(nn.Module):
#     """Transolver model for point clouds."""
#     num_layers: int
#     num_slices: int
#     num_heads: int
#     hidden_dim: int
#     num_channels: int
#     mlp_dim: int
#     output_dim: int
#     dropout_rate: float = 0.0
#     aggregation: str = "mean"
    
#     # Three options:
#     # 1. aggregate at point level
#     # 2. aggregate at slice level
#     # 3. have global token

#     @nn.compact
#     def __call__(self, x, train: bool = True):

#         x = Transolver(
#             num_layers=self.num_layers,
#             num_slices=self.num_slices,
#             num_heads=self.num_heads,
#             hidden_dim=self.hidden_dim,
#             num_channels=self.num_channels,
#             mlp_dim=self.mlp_dim,
#             output_dim=self.output_dim,
#             dropout_rate=self.dropout_rate
#         )(x, train=train,)
#         if self.aggregation == "mean":
#             return jnp.mean(x, axis=1)
#         elif self.aggregation == "max":
#             return jnp.max(x, axis=1)
#         elif self.aggregation == "attention":

#         else:
#             raise ValueError(f"Invalid aggregation method: {self.aggregation}")
