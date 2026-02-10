"""Transformer Autoencoder for learning page embeddings from pageview patterns."""

import torch
import torch.nn as nn

from .positional_encoding import PositionalEncoding


class TransformerAutoencoder(nn.Module):
    """
    Transformer-based Autoencoder for learning compact representations of
    Wikipedia pageview time series.

    Architecture:
    - Input projection: [batch, seq_len, 1] -> [batch, seq_len, d_model]
    - Positional encoding: Sinusoidal
    - Encoder: N transformer encoder layers
    - Bottleneck: Mean pooling -> embedding
    - Decoder: N transformer decoder layers with learnable position queries
    - Output projection: [batch, seq_len, d_model] -> [batch, seq_len]
    """

    def __init__(
        self,
        seq_length: int = 2160,
        d_model: int = 128,
        n_heads: int = 8,
        d_ff: int = 512,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 4,
        dropout: float = 0.1,
        embedding_dim: int = 128,
    ):
        """
        Initialize the Transformer Autoencoder.

        Args:
            seq_length: Length of input sequences (90 days * 24 hours = 2160)
            d_model: Dimension of the model
            n_heads: Number of attention heads
            d_ff: Dimension of feedforward layers
            n_encoder_layers: Number of encoder layers
            n_decoder_layers: Number of decoder layers
            dropout: Dropout rate
            embedding_dim: Dimension of the bottleneck embedding
        """
        super().__init__()

        self.seq_length = seq_length
        self.d_model = d_model
        self.embedding_dim = embedding_dim

        # Input projection: [batch, seq_len, 1] -> [batch, seq_len, d_model]
        self.input_projection = nn.Linear(1, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_length, dropout=dropout)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for better training stability
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_encoder_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Bottleneck: project encoded representation to embedding
        # We'll use mean pooling followed by a linear projection
        self.bottleneck_projection = nn.Linear(d_model, embedding_dim)

        # Decoder: expand embedding back to sequence
        # Learnable position queries for the decoder
        self.decoder_queries = nn.Parameter(torch.randn(1, seq_length, d_model) * 0.02)
        self.pos_decoder = PositionalEncoding(d_model, max_len=seq_length, dropout=dropout)

        # Embedding expansion for decoder
        self.embedding_expansion = nn.Linear(embedding_dim, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=n_decoder_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Output projection: [batch, seq_len, d_model] -> [batch, seq_len]
        self.output_projection = nn.Linear(d_model, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for name, param in self.named_parameters():
            if param.dim() > 1 and "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input sequence to embedding.

        Args:
            x: Input tensor [batch, seq_len] or [batch, seq_len, 1]

        Returns:
            Embedding tensor [batch, embedding_dim]
        """
        # Ensure input is [batch, seq_len, 1]
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        # Project input to model dimension
        x = self.input_projection(x)  # [batch, seq_len, d_model]

        # Add positional encoding
        x = self.pos_encoder(x)

        # Encode
        encoded = self.encoder(x)  # [batch, seq_len, d_model]

        # Mean pooling over sequence dimension
        pooled = encoded.mean(dim=1)  # [batch, d_model]

        # Project to embedding dimension
        embedding = self.bottleneck_projection(pooled)  # [batch, embedding_dim]

        return embedding

    def decode(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Decode embedding back to sequence.

        Args:
            embedding: Embedding tensor [batch, embedding_dim]

        Returns:
            Reconstructed sequence [batch, seq_len]
        """
        batch_size = embedding.size(0)

        # Expand embedding to memory for cross-attention
        memory = self.embedding_expansion(embedding)  # [batch, d_model]
        memory = memory.unsqueeze(1).expand(-1, self.seq_length, -1)  # [batch, seq_len, d_model]

        # Get learnable position queries and expand for batch
        queries = self.decoder_queries.expand(batch_size, -1, -1)  # [batch, seq_len, d_model]

        # Add positional encoding to queries
        queries = self.pos_decoder(queries)

        # Decode using cross-attention between queries and memory
        decoded = self.decoder(queries, memory)  # [batch, seq_len, d_model]

        # Project to output dimension
        output = self.output_projection(decoded).squeeze(-1)  # [batch, seq_len]

        return output

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode then decode.

        Args:
            x: Input tensor [batch, seq_len] or [batch, seq_len, 1]

        Returns:
            Tuple of (reconstructed sequence [batch, seq_len], embedding [batch, embedding_dim])
        """
        embedding = self.encode(x)
        reconstructed = self.decode(embedding)
        return reconstructed, embedding

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get embedding for input sequence without decoding.

        Args:
            x: Input tensor [batch, seq_len] or [batch, seq_len, 1]

        Returns:
            Embedding tensor [batch, embedding_dim]
        """
        return self.encode(x)

    @property
    def num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"seq_length={self.seq_length}, "
            f"d_model={self.d_model}, "
            f"embedding_dim={self.embedding_dim}, "
            f"params={self.num_parameters:,})"
        )
