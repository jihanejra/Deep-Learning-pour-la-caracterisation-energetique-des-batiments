"""Architectures PyTorch : CNN-GRU et Transformer à patches.

Les deux modèles ont la même interface : une séquence ``(B, C, L)`` en entrée et
``n_out`` sorties standardisées. Ils se terminent par la même tête MLP, ce qui rend
leur comparaison équitable.
"""

from __future__ import annotations

import torch
import torch.nn as nn

MODELS: tuple[str, ...] = ("cnn_gru", "transformer")


def _head(d_in: int, n_out: int, dropout: float, hidden: int = 64) -> nn.Sequential:
    """Tête de régression commune aux deux modèles."""
    return nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_out))


class CNNGRURegressor(nn.Module):
    """CNN-GRU : architecture de l'article de référence (Nassif et al.).

    Trois blocs convolutifs extraient des motifs locaux et divisent la longueur par 8 ;
    un GRU bidirectionnel à deux couches résume ensuite la séquence. Les états finaux
    des deux sens sont concaténés et envoyés dans la tête MLP.
    """

    def __init__(self, in_channels: int = 2, channels: int = 64, kernel: int = 5, n_conv: int = 3,
                 hidden: int = 64, n_gru_layers: int = 2, dropout: float = 0.1, n_out: int = 5) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        c_in = in_channels
        for _ in range(n_conv):
            layers += [nn.Conv1d(c_in, channels, kernel, padding=kernel // 2),
                       nn.BatchNorm1d(channels), nn.GELU(), nn.MaxPool1d(2)]
            c_in = channels
        self.cnn = nn.Sequential(*layers)
        self.gru = nn.GRU(input_size=channels, hidden_size=hidden, num_layers=n_gru_layers,
                          batch_first=True, bidirectional=True, dropout=dropout)
        self.head = _head(2 * hidden, n_out, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.cnn(x).transpose(1, 2)                         # (B, L / 8, channels)
        _, h_n = self.gru(h)                                    # (2 * couches, B, hidden)
        return self.head(torch.cat([h_n[-2], h_n[-1]], dim=1))  # états finaux aller + retour


class PatchTransformerRegressor(nn.Module):
    """Transformer à patches : la séquence est découpée en patches, chacun devenant un token.

    Un token de synthèse ``[CLS]`` est ajouté en tête et un encodage de position est appris.
    Il n'y a pas de masque causal. Le vecteur ``[CLS]`` final alimente la tête MLP.
    """

    def __init__(self, in_channels: int = 2, seq_len: int = 480, patch: int = 16, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 4, d_ff: int = 128, dropout: float = 0.1,
                 n_out: int = 5) -> None:
        super().__init__()
        if seq_len % patch != 0:
            raise ValueError("La longueur de la séquence doit être un multiple de la taille des patches")
        self.patch = patch
        self.n_patches = seq_len // patch
        self.embed = nn.Linear(in_channels * patch, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches + 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = _head(d_model, n_out, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.shape
        x = x.reshape(b, c, self.n_patches, self.patch).permute(0, 2, 1, 3)
        tokens = self.embed(x.reshape(b, self.n_patches, c * self.patch))       # (B, P, d)
        tokens = torch.cat([self.cls.expand(b, -1, -1), tokens], dim=1) + self.pos
        return self.head(self.norm(self.encoder(tokens)[:, 0]))


def _init_he(layer: nn.Module) -> None:
    """Initialisation de He pour les convolutions et les couches linéaires."""
    if isinstance(layer, (nn.Linear, nn.Conv1d)):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


def build_model(name: str, in_channels: int, seq_len: int = 480, n_out: int = 5,
                dropout: float = 0.1) -> nn.Module:
    """Crée un modèle par son nom.

    Le CNN-GRU reçoit une initialisation de He ; le Transformer garde
    l'initialisation par défaut de PyTorch, mieux adaptée aux couches d'attention.

    Args:
        name: ``"cnn_gru"`` ou ``"transformer"``.
        in_channels: nombre de canaux d'entrée.
        seq_len: longueur des séquences.
        n_out: nombre de sorties.
        dropout: taux de dropout.
    """
    if name == "cnn_gru":
        model: nn.Module = CNNGRURegressor(in_channels, dropout=dropout, n_out=n_out)
        model.apply(_init_he)
    elif name == "transformer":
        model = PatchTransformerRegressor(in_channels, seq_len=seq_len, dropout=dropout, n_out=n_out)
    else:
        raise ValueError(f"Modèle inconnu : {name!r} (attendu : {MODELS})")
    return model


def count_parameters(model: nn.Module) -> int:
    """Nombre de paramètres entraînables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
