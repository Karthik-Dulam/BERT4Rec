"""
In this file we will implement the BERT Transformer model using PyTorch.
"""

import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, input_size: int, num_heads: int):
        super().__init__()
        assert input_size % num_heads == 0
        self.query = nn.Linear(input_size, input_size)
        self.key = nn.Linear(input_size, input_size)
        self.value = nn.Linear(input_size, input_size)
        self.dropout_trans = nn.Dropout(0.1)
        self.pffn = nn.Sequential(
            nn.Linear(input_size, 4 * input_size),
            nn.GELU(),
            nn.Linear(4 * input_size, input_size),
        )
        self.dropout_pffn = nn.Dropout(0.1)
        self.num_heads = num_heads

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, input_size = tensor.size()

        d_tensor = input_size // self.num_heads
        tensor = tensor.view(batch_size, length, self.num_heads, d_tensor).transpose(
            1, 2
        )
        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor

    def forward(self, x, mask=None):
        q, k, v = self.query(x), self.key(x), self.value(x)
        q, k, v = self.split(q), self.split(k), self.split(v)
        score = q @ k.transpose(1, 2) / torch.sqrt(k.size(-1))
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        attention = nn.softmax(score) @ v
        out = self.concat(attention)
        out = self.dropout(out)
        x = x + out
        x = nn.LayerNorm(x)
        x = x + self.dropout_pffn(self.pffn(x))
        x = nn.LayerNorm(x)
        return x


class Embedding(nn.Module):
    """
    Embedding with positional encoding
    """

    def __init__(self, max_len: int, input_size: int):
        super().__init__()
        self.embedding = nn.Embedding(max_len, input_size)
        self.positional_encoding = nn.Parameter(torch.zeros(max_len, input_size))


class BERT(nn.Module):
    def __init__(self, input_size: int, num_heads: int, num_layers: int):
        super().__init__()
        self.transformers = nn.ModuleList(
            [Transformer(input_size, num_heads) for _ in range(num_layers)]
        )
        self.fc = nn.Linear(input_size, 2)

    def forward(self, x):
        for transformer in self.transformers:
            x = transformer(x)
        x = self.fc(x)
        return x
