"""CNN Autoencoder for learning page embeddings from pageview patterns."""

import torch
import torch.nn as nn


class CNNAutoencoder(nn.Module):
    """
    1D CNN-based Autoencoder for learning compact representations of
    Wikipedia pageview time series.

    Architecture:
    - Encoder: Strided 1D convolutions with BatchNorm and GELU activations
      [batch, 1, 2160] -> ... -> [batch, channels[-1], compressed] -> [batch, embedding_dim]
    - Decoder: Transposed convolutions mirroring the encoder
      [batch, embedding_dim] -> [batch, channels[-1], compressed] -> ... -> [batch, 1, 2160]
    """

    def __init__(
        self,
        seq_length: int = 2160,
        channels: tuple[int, ...] = (32, 64, 128, 256, 256),
        embedding_dim: int = 128,
        dropout: float = 0.1,
    ):
        """
        Initialize the CNN Autoencoder.

        Args:
            seq_length: Length of input sequences (90 days * 24 hours = 2160)
            channels: Number of channels at each encoder stage
            embedding_dim: Dimension of the bottleneck embedding
            dropout: Dropout rate
        """
        super().__init__()

        self.seq_length = seq_length
        self.channels = channels
        self.embedding_dim = embedding_dim

        # Build encoder
        encoder_layers = []
        in_ch = 1
        for i, out_ch in enumerate(channels):
            if i == 0:
                # First layer: larger kernel to capture broad patterns
                encoder_layers.append(
                    nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=2, padding=3)
                )
            else:
                encoder_layers.append(
                    nn.Conv1d(in_ch, out_ch, kernel_size=5, stride=2, padding=2)
                )
            encoder_layers.append(nn.BatchNorm1d(out_ch))
            encoder_layers.append(nn.GELU())
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
            in_ch = out_ch

        self.encoder_conv = nn.Sequential(*encoder_layers)
        self.encoder_pool = nn.AdaptiveAvgPool1d(1)
        self.bottleneck_projection = nn.Linear(channels[-1], embedding_dim)

        # Compute compressed length for decoder input
        # Run a dummy forward pass to determine the spatial size before pooling
        with torch.no_grad():
            dummy = torch.zeros(1, 1, seq_length)
            dummy_out = self.encoder_conv(dummy)
            self._compressed_len = dummy_out.shape[2]

        # Build decoder
        self.embedding_expansion = nn.Linear(
            embedding_dim, channels[-1] * self._compressed_len
        )

        decoder_layers = []
        rev_channels = list(reversed(channels))
        # Output paddings needed to match encoder's spatial dimensions exactly
        # Encoder: 2160 -> 1080 -> 540 -> 270 -> 135 -> 68
        # Decoder: 68 -> 135 -> 270 -> 540 -> 1080 -> 2160
        output_paddings = self._compute_output_paddings(seq_length, len(channels))

        for i in range(len(rev_channels) - 1):
            in_ch = rev_channels[i]
            out_ch = rev_channels[i + 1]
            if i == len(rev_channels) - 2:
                # Second-to-last transposed conv (mirrors first encoder conv)
                decoder_layers.append(
                    nn.ConvTranspose1d(
                        in_ch,
                        out_ch,
                        kernel_size=5,
                        stride=2,
                        padding=2,
                        output_padding=output_paddings[i],
                    )
                )
            else:
                decoder_layers.append(
                    nn.ConvTranspose1d(
                        in_ch,
                        out_ch,
                        kernel_size=5,
                        stride=2,
                        padding=2,
                        output_padding=output_paddings[i],
                    )
                )
            decoder_layers.append(nn.BatchNorm1d(out_ch))
            decoder_layers.append(nn.GELU())
            if dropout > 0:
                decoder_layers.append(nn.Dropout(dropout))

        # Final layer: back to 1 channel, no activation (regression output)
        decoder_layers.append(
            nn.ConvTranspose1d(
                rev_channels[-1],
                1,
                kernel_size=7,
                stride=2,
                padding=3,
                output_padding=output_paddings[-1],
            )
        )

        self.decoder_conv = nn.Sequential(*decoder_layers)

        # Initialize weights
        self._init_weights()

    def _compute_output_paddings(
        self, seq_length: int, n_layers: int
    ) -> list[int]:
        """Compute output_padding values for each transposed conv layer."""
        # Forward pass sizes through encoder
        sizes = [seq_length]
        L = seq_length
        for i in range(n_layers):
            if i == 0:
                k, s, p = 7, 2, 3
            else:
                k, s, p = 5, 2, 2
            L = (L + 2 * p - k) // s + 1
            sizes.append(L)

        # Reverse to get target sizes for decoder
        # sizes = [2160, 1080, 540, 270, 135, 68]
        # Decoder goes: 68 -> 135 -> 270 -> 540 -> 1080 -> 2160
        target_sizes = list(reversed(sizes))
        # target_sizes[0] is the compressed size (68), we start from there
        # We need output_paddings for layers going from target_sizes[0] -> target_sizes[1], etc.

        output_paddings = []
        for i in range(len(target_sizes) - 1):
            src = target_sizes[i]
            tgt = target_sizes[i + 1]
            if i == len(target_sizes) - 2:
                # Last deconv layer mirrors first encoder layer (k=7, s=2, p=3)
                k, s, p = 7, 2, 3
            else:
                k, s, p = 5, 2, 2
            # ConvTranspose1d: out = (in - 1) * s - 2*p + k + output_padding
            base_out = (src - 1) * s - 2 * p + k
            op = tgt - base_out
            output_paddings.append(op)

        return output_paddings

    def _init_weights(self):
        """Initialize weights using Kaiming initialization (suitable for ReLU/GELU)."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input sequence to embedding.

        Args:
            x: Input tensor [batch, seq_len] or [batch, 1, seq_len]

        Returns:
            Embedding tensor [batch, embedding_dim]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, seq_len]

        x = self.encoder_conv(x)  # [batch, channels[-1], compressed_len]
        x = self.encoder_pool(x)  # [batch, channels[-1], 1]
        x = x.squeeze(-1)  # [batch, channels[-1]]
        embedding = self.bottleneck_projection(x)  # [batch, embedding_dim]

        return embedding

    def decode(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Decode embedding back to sequence.

        Args:
            embedding: Embedding tensor [batch, embedding_dim]

        Returns:
            Reconstructed sequence [batch, seq_len]
        """
        x = self.embedding_expansion(embedding)  # [batch, channels[-1] * compressed_len]
        x = x.view(
            x.size(0), self.channels[-1], self._compressed_len
        )  # [batch, channels[-1], compressed_len]
        x = self.decoder_conv(x)  # [batch, 1, seq_len]
        return x.squeeze(1)  # [batch, seq_len]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode then decode.

        Args:
            x: Input tensor [batch, seq_len] or [batch, 1, seq_len]

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
            x: Input tensor [batch, seq_len] or [batch, 1, seq_len]

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
            f"channels={self.channels}, "
            f"embedding_dim={self.embedding_dim}, "
            f"params={self.num_parameters:,})"
        )
