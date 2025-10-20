import torch
import torch.nn as nn
import math
import logging

logger = logging.getLogger(__name__)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.qkv_layer = nn.Linear(d_model, 3 * d_model)
        self.out_layer = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv_layer(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        context = torch.matmul(attention, v)
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.out_layer(context)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
    def forward(self, x):
        return self.net(x)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.dropout(self.attention(x, mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x

class TransformerFeatureExtractor(nn.Module):
    """
    A simplified and consolidated Transformer architecture for supervised feature extraction.
    It takes a single sequence of raw market data and outputs a rich feature vector.
    """
    def __init__(self,
                 input_dim: int,
                 d_model: int,
                 n_heads: int,
                 dim_feedforward: int,
                 num_layers: int,
                 output_dim: int,
                 dropout: float = 0.1,
                 max_seq_len: int = 100):
        super().__init__()
        self.d_model = d_model
        self.output_dim = output_dim

        self.input_projection = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        self.positional_encoding = nn.Parameter(torch.randn(max_seq_len, d_model) * 0.02)

        self.transformer_layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.LayerNorm(d_model // 2),
            nn.Linear(d_model // 2, output_dim)
        )

        self._init_weights()
        logger.info(f"Initialized TransformerFeatureExtractor with output_dim={output_dim}")

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.constant_(self.input_projection.bias, 0)
        for layer in self.output_projection:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x = self.input_projection(x)
        x = self.input_norm(x)
        pos_enc = self.positional_encoding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        x = x + pos_enc
        for layer in self.transformer_layers:
            x = layer(x)
        x_pooled = x.mean(dim=1)
        final_features = self.output_projection(x_pooled)
        return final_features
