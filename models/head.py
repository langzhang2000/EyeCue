import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, hidden_dim=256, dropout_prob=0.1):
        """
        Binary classification head.

        `fc1` is created lazily on the first forward pass, because its input
        dimension depends on the number of fused tokens T (T * D).
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = None                      # lazily built as Linear(T*D -> hidden_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, T, D]
        Returns:
            logits: Tensor of shape [B, 2]
        """
        B, T, D = x.shape
        x = x.reshape(B, T * D)         # flatten -> [B, T*D]

        # Lazily build the first layer (T*D -> hidden_dim)
        if self.fc1 is None:
            input_dim = T * D
            self.fc1 = nn.Linear(input_dim, self.hidden_dim).to(x.device)

        # Hidden layer + activation + dropout
        x = self.fc1(x)                 # -> [B, hidden_dim]
        x = self.activation(x)
        x = self.dropout(x)

        logits = self.fc2(x)            # -> [B, 2]
        return logits
