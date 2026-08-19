"""
Evaluation model: TCN + Attention neural network for market state scoring.

Implements StockEvalNet, a Temporal Convolutional Network with multi-head
self-attention that evaluates market states and outputs a Position_Score
in [-1.0, +1.0].

Also implements ModelManager for production model lifecycle: loading with
checksum validation, hot-swap (atomic model replacement), version tracking,
GPU/CPU device management, and high-level predict() API with OOM retry.

Architecture:
    Input (batch, lookback=60, num_features=63)
    → Transpose to (batch, 63, 60)
    → TCN Block 1: 63→128, kernel=3, dilation=1
    → TCN Block 2: 128→128, kernel=3, dilation=2
    → TCN Block 3: 128→64, kernel=3, dilation=4
    → Transpose to (batch, 60, 64)
    → MultiheadAttention (4 heads, embed_dim=64) + residual + LayerNorm
    → Transpose to (batch, 64, 60)
    → AdaptiveAvgPool1d → (batch, 64)
    → Linear(64, 1) → Tanh → output in [-1, 1]

Target parameter count: ~180K (within ±20K).
"""

import logging

import torch
import torch.nn as nn

from engine.config import ModelConfig

logger = logging.getLogger(__name__)


class TCNBlock(nn.Module):
    """
    Temporal Convolutional Block with dilated causal convolutions and residual connection.

    Structure:
        Conv1d (causal, dilated) → BatchNorm → GELU → Dropout
        → Conv1d (causal, dilated) → BatchNorm → Residual add
        → GELU

    Uses causal padding to prevent information leakage from future timesteps.
    If in_channels != out_channels, a 1x1 convolution adapts the residual path.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Causal padding: (kernel_size - 1) * dilation on the left side only
        self.causal_padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation, padding=self.causal_padding
        )
        self.norm1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, dilation=dilation, padding=self.causal_padding
        )
        self.norm2 = nn.BatchNorm1d(out_channels)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Residual connection: 1x1 conv if channel dimensions differ, else identity
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, seq_len)

        Returns:
            (batch, out_channels, seq_len)
        """
        seq_len = x.size(-1)
        res = self.residual(x)

        # First conv block: Conv1d → trim causal → BatchNorm → GELU → Dropout
        out = self.conv1(x)
        out = out[..., :seq_len]  # Trim causal padding to maintain seq_len
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)

        # Second conv block: Conv1d → trim causal → BatchNorm
        out = self.conv2(out)
        out = out[..., :seq_len]  # Trim causal padding
        out = self.norm2(out)

        # Residual add + final activation
        out = out + res
        out = self.activation(out)

        return out



# Module-level registry: theo dõi instance StockEvalNet cuối cùng được tạo
# Dùng cho retrain flow — cho phép _retrain_model() train đúng model instance
# mà caller đang sử dụng.
_active_model_instance: "StockEvalNet | None" = None


def get_active_model() -> "StockEvalNet | None":
    """Trả về StockEvalNet instance được tạo gần nhất (hoặc None)."""
    return _active_model_instance


class StockEvalNet(nn.Module):
    """
    TCN + Attention model for market state evaluation.

    Evaluates a market state represented as a feature tensor and outputs
    a Position_Score in [-1.0, +1.0].

    Parameters: ~180K (well within 4GB VRAM constraint)
    Inference: ~5ms per sample on RTX 2060
    """

    def __init__(self, config: ModelConfig = None):
        super().__init__()
        if config is None:
            config = ModelConfig()

        # Đăng ký instance này vào module-level registry
        global _active_model_instance
        _active_model_instance = self

        self.config = config
        num_features = config.num_features
        tcn_channels = config.tcn_channels
        dilations = config.dilations
        kernel_size = config.kernel_size
        attention_heads = config.attention_heads
        attention_dim = config.attention_dim
        dropout = config.dropout

        # Validate: last TCN output channel must match attention embed_dim
        if tcn_channels[-1] != attention_dim:
            raise ValueError(
                f"Last TCN channel ({tcn_channels[-1]}) must equal "
                f"attention_dim ({attention_dim})"
            )

        # Build TCN backbone
        # Input channels: num_features (63)
        # Channel progression: 63 → 128 → 128 → 64
        tcn_layers = []
        in_ch = num_features
        for out_ch, dil in zip(tcn_channels, dilations):
            tcn_layers.append(
                TCNBlock(in_ch, out_ch, kernel_size=kernel_size, dilation=dil, dropout=dropout)
            )
            in_ch = out_ch
        self.tcn = nn.Sequential(*tcn_layers)

        # Multi-head self-attention
        # embed_dim = last TCN channel = attention_dim = 64
        self.attention = nn.MultiheadAttention(
            embed_dim=attention_dim, num_heads=attention_heads, batch_first=True, dropout=dropout
        )
        self.layer_norm = nn.LayerNorm(attention_dim)

        # Output head: AdaptiveAvgPool1d → Flatten → Linear → Tanh
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(attention_dim, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, lookback, num_features)

        Returns:
            Position scores of shape (batch, 1) with values in [-1.0, +1.0]
        """
        # Transpose for Conv1d: (batch, num_features, lookback)
        x = x.transpose(1, 2)

        # TCN backbone: (batch, 64, lookback)
        x = self.tcn(x)

        # Transpose for attention: (batch, lookback, 64)
        x = x.transpose(1, 2)

        # Multi-head self-attention with residual connection
        attn_out, _ = self.attention(x, x, x)
        x = self.layer_norm(x + attn_out)

        # Transpose for pooling: (batch, 64, lookback)
        x = x.transpose(1, 2)

        # Global average pooling: (batch, 64, 1) → (batch, 64)
        x = self.pool(x)
        x = x.squeeze(-1)

        # Linear head with Tanh: (batch, 1)
        x = self.head(x)

        return x

    def count_parameters(self) -> int:
        """Count total trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Re-export ModelManager so existing imports from engine.evaluation_model still work
from engine.model_manager import ModelManager, _compute_state_dict_checksum  # noqa: E402, F401

__all__ = ["TCNBlock", "StockEvalNet", "ModelManager", "_compute_state_dict_checksum", "get_active_model"]
