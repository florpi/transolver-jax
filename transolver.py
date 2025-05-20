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

        qkv = nn.Dense(3 * self.num_heads * self.dim_per_head)(x)  # [B, N, 3 * H * D]
        qkv = qkv.reshape(
            batch_size, num_points, 3, self.num_heads, self.dim_per_head
        )  # [B, N, 3, H, D]
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))  # [3, B, H, N, D]
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, H, N, D]

        # Aggregating points to physics-aware tokens (Eq. 2)
        # First reshape to [B, N, M, 1, 1]
        slice_weights_expanded = slice_weights[..., None, None]
        # Then transpose to [B, 1, M, N, 1] to align with head dimension
        slice_weights_for_qkv = jnp.transpose(
            slice_weights_expanded, (0, 3, 2, 1, 4)
        )  # [B, 1, M, N, 1]
        weight_sums = jnp.sum(
            slice_weights_for_qkv, axis=3, keepdims=True
        )  # [B, 1, M, 1, 1]
        weight_sums = weight_sums.reshape(
            batch_size, 1, self.num_slices, 1
        )  # [B, 1, M, 1]

        q_sliced = jnp.sum(q[:, :, None, :, :] * slice_weights_for_qkv, axis=3)
        k_sliced = jnp.sum(k[:, :, None, :, :] * slice_weights_for_qkv, axis=3)
        v_sliced = jnp.sum(v[:, :, None, :, :] * slice_weights_for_qkv, axis=3)

        # Normalize by the sum of slice weights
        q_sliced = q_sliced / (
            weight_sums.reshape(batch_size, 1, self.num_slices, 1) + 1e-9
        )
        k_sliced = k_sliced / (
            weight_sums.reshape(batch_size, 1, self.num_slices, 1) + 1e-9
        )
        v_sliced = v_sliced / (
            weight_sums.reshape(batch_size, 1, self.num_slices, 1) + 1e-9
        )

        scale = jnp.sqrt(self.dim_per_head)
        attention = jnp.matmul(q_sliced, jnp.transpose(k_sliced, (0, 1, 3, 2))) / scale
        attention = jax.nn.softmax(attention, axis=-1)  # [B, H, M, M]

        token_output = jnp.matmul(attention, v_sliced)  # [B, H, M, D]
        token_output = jnp.transpose(token_output, (0, 2, 1, 3))[:, None, ...]

        output = jnp.sum(token_output * slice_weights_expanded, axis=2)  # [B, N, H, D]
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