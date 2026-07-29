import torch
import torch.nn as nn


class GazeEncoder(nn.Module):
    def __init__(self, input_dim=2, embed_dim=768, num_heads=8, num_layers=1, dropout=0.0):
        """
        Lightweight transformer encoder over raw (x, y) gaze coordinates.

        A learnable CLS token is prepended to the projected gaze sequence, so the
        encoder produces both a pooled representation and per-frame gaze tokens.
        """
        super().__init__()
        self.embed_dim = embed_dim

        # 1) Project (x, y) into the D-dimensional token space
        self.fc = nn.Linear(input_dim, embed_dim)
        self.fc_norm = nn.LayerNorm(embed_dim)

        # 2) Learnable CLS token, shape [1, 1, D]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 3) Stacked MultiheadAttention + LayerNorm + Dropout
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'mh': nn.MultiheadAttention(embed_dim, num_heads, batch_first=True),
                'norm': nn.LayerNorm(embed_dim),
                'dropout': nn.Dropout(dropout)
            })
            for _ in range(num_layers)
        ])

        # Parameter initialization
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.fc.weight, std=0.02)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)

    def forward(self, gaze_xy):
        """
        Args:
            gaze_xy: Tensor of shape [B, Q, 2]
        Returns:
            cls_out:     Tensor of shape [B, 1, D]
            gaze_tokens: Tensor of shape [B, Q, D]
        """
        B, Q, _ = gaze_xy.shape

        # 1) Project and normalize
        x = self.fc(gaze_xy)           # [B, Q, D]
        x = self.fc_norm(x)

        # 2) Prepend the CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat((cls_tokens, x), dim=1)          # [B, Q+1, D]

        # 3) Self-attention + Norm + Dropout, layer by layer
        for layer in self.layers:
            attn_out, _ = layer['mh'](x, x, x)
            x = layer['norm'](attn_out)
            x = layer['dropout'](x)

        # 4) Split the CLS token off from the gaze tokens
        cls_out = x[:, :1, :]        # [B, 1, D]
        gaze_tokens = x[:, 1:, :]    # [B, Q, D]

        return cls_out, gaze_tokens
