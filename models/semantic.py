import torch
import torch.nn as nn


class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        """
        One "cross-attention + residual + LayerNorm + FFN + residual + LayerNorm" block.

        Args:
            embed_dim: feature dimension E of the patch and gaze tokens.
            num_heads: number of attention heads.
            dropout:   dropout rate used in the attention and the FFN.
        """
        super().__init__()
        # 1. Cross-attention: gaze tokens query the selected patch tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # inputs/outputs are (B, L, E)
        )
        # 2. First residual + LayerNorm (applied after adding the attention output)
        self.norm1 = nn.LayerNorm(embed_dim)

        # 3. Feed-forward network: two linear layers + activation + dropout
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),  # expand to 4*E
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),  # project back to E
            nn.Dropout(dropout),
        )
        # 4. Second residual + LayerNorm (applied after adding the FFN output)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, gaze: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, N_patch, E) patch tokens fed into this layer
            gaze: (B, N_gaze,  E) gaze tokens (kept fixed across layers)
        Returns:
            out:  (B, N_patch, E) patch tokens after cross-attention + FFN
        """
        # ---- Cross-attention ----
        # Gaze tokens act as the query; patch tokens are the key and value.
        attn_out, _ = self.cross_attn(
            query=gaze,   # (B, N_gaze,  E)
            key=x,        # (B, N_patch, E)
            value=x       # (B, N_patch, E)
        )
        # Residual + LayerNorm
        x = x + attn_out       # (B, N_patch, E)
        x = self.norm1(x)      # (B, N_patch, E)

        # ---- Feed-forward ----
        ffn_out = self.ffn(x)  # (B, N_patch, E)
        x = x + ffn_out        # residual
        x = self.norm2(x)      # (B, N_patch, E)

        return x


class Semantic(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        num_heads=8,
        dropout: float = 0.1,
        num_layers: int = 2
    ):
        """
        Semantic fusion module: a stack of `num_layers` CrossAttentionBlocks.

        Args:
            embed_dim:  feature dimension E of the patch/gaze tokens.
            num_heads:  number of attention heads.
            dropout:    dropout rate used in the attention and the FFN.
            num_layers: how many CrossAttentionBlocks to stack.
        """
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionBlock(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, patch_tokens: torch.Tensor, gaze_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tokens: (B, N_patch, E) gaze-selected patch tokens
            gaze_tokens:  (B, N_gaze,  E) gaze tokens used as the query

        Returns:
            out: (B, 1, E) mean-pooled fused representation
        """
        x = patch_tokens
        for block in self.layers:
            x = block(x, gaze_tokens)
        x = x.mean(dim=1, keepdim=True)
        return x
