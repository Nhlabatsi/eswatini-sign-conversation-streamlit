
"""
model.py
========
    landmark sequence (B, T, F)
        -> LandmarkSequenceEncoder   (per-frame projection + positional encoding)
        -> TemporalConvNet (TCN)     (dilated causal 1D convs, growing receptive field)
        -> ClassifierHead            (masked temporal pooling + linear)
        -> logits (B, num_classes)
All pieces are plain nn.Module subclasses so you can swap any one of
them out independently (e.g. try a GRU encoder, or a Transformer TCN
replacement) without touching the rest.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


# ---------------------------------------------------------------------------
# 1. Landmark sequence encoder
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding, added to the projected features."""

    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        T = x.size(1)
        return x + self.pe[:, :T, :]


class LandmarkSequenceEncoder(nn.Module):
    """
    Projects raw per-frame landmark vectors into a d_model-dim embedding
    space, with LayerNorm + dropout + positional encoding. Frames that
    are entirely zero (e.g. both hands missing) are left as-is; the
    projection layer learns to handle that implicitly.
    """

    def __init__(self, input_dim: int, d_model: int = 256, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_enc = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_dim) -> (B, T, d_model)
        x = self.proj(x)
        x = self.pos_enc(x)
        return self.dropout(x)


# ---------------------------------------------------------------------------
# 2. Temporal Convolutional Network (Bai et al., 2018 style)
# ---------------------------------------------------------------------------

class Chomp1d(nn.Module):
    """Removes the extra right-padding added to keep convolutions causal."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Two causal dilated conv1d layers with weight norm, ReLU, dropout,
    and a residual connection (with a 1x1 conv if channel dims differ).
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(out_channels, out_channels, kernel_size,
                      padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2,
        )
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        )
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    Stack of TemporalBlocks with exponentially growing dilation
    (1, 2, 4, 8, ...), so the receptive field grows quickly with depth
    while staying fully causal (good for eventual streaming/live use).
    """

    def __init__(self, num_inputs: int, num_channels: list[int], kernel_size: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout=dropout)
            )
        self.network = nn.Sequential(*layers)
        self.receptive_field = 1 + 2 * (kernel_size - 1) * (2 ** num_levels - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T) -> (B, C_out, T)
        return self.network(x)


# ---------------------------------------------------------------------------
# 3. Classifier head
# ---------------------------------------------------------------------------

def masked_mean_max_pool(x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
    """
    Shared pooling helper: masked mean+max over the time axis.
    x: (B, C, T) -> returns (B, 2C)
    """
    B, C, T = x.shape
    if lengths is None:
        mask = torch.ones(B, T, dtype=torch.bool, device=x.device)
    else:
        arange = torch.arange(T, device=x.device).unsqueeze(0)
        mask = arange < lengths.unsqueeze(1)

    mask_f = mask.unsqueeze(1).float()
    summed = (x * mask_f).sum(dim=2)
    counts = mask_f.sum(dim=2).clamp(min=1.0)
    mean_pool = summed / counts

    x_masked_for_max = x.masked_fill(~mask.unsqueeze(1), float("-inf"))
    max_pool, _ = x_masked_for_max.max(dim=2)
    max_pool = torch.nan_to_num(max_pool, neginf=0.0)

    return torch.cat([mean_pool, max_pool], dim=1)


class ClassifierHead(nn.Module):
    """
    Masked mean+max temporal pooling over the TCN output, followed by
    an MLP classifier. Masking matters because padded timesteps
    shouldn't influence the pooled representation.
    """

    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_channels, num_classes),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        pooled = masked_mean_max_pool(x, lengths)  # (B, 2C)
        return self.fc(pooled)


# ---------------------------------------------------------------------------
# 4. Full model
# ---------------------------------------------------------------------------

class ArcMarginProduct(nn.Module):
    """
    Additive Angular Margin head (ArcFace, Deng et al. 2019).
    Standard softmax classifiers struggle badly when the number of
    classes is very large relative to samples per class (often just 1,
    as with a sign-language dictionary of ~1600 words with one video
    each) -- this is the same regime as face recognition (millions of
    identities, 1-2 photos each), which is exactly why the field moved
    to angular-margin losses instead of plain softmax.
    Instead of a raw linear classifier, this normalizes both the
    embedding and each class's weight vector onto a unit hypersphere,
    computes cosine similarity, and adds an angular margin to the
    true class during training -- forcing embeddings of the same
    class to cluster tightly and different classes to spread apart
    by at least `margin` radians. This is dramatically more stable to
    optimize at high class counts / few samples per class than a
    plain nn.Linear + softmax.
    """

    def __init__(self, in_features: int, num_classes: int, scale: float = 30.0,
                 margin: float = 0.3, easy_margin: bool = False):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin

        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        self._set_margin(margin)

    def _set_margin(self, margin: float):
        """(Re)computes the margin and its derived constants. Used both at
        init and by set_margin() for margin warm-up during training."""
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def set_margin(self, margin: float):
        """Update the angular margin mid-training (for margin warm-up:
        start at 0 and ramp to the target value over the first N epochs,
        which tends to speed up early convergence at very large class
        counts by not fighting the margin constraint before the
        embedding space has any structure yet)."""
        self._set_margin(margin)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        emb_norm = F.normalize(embeddings, dim=1)
        w_norm = F.normalize(self.weight, dim=1)
        cosine = emb_norm @ w_norm.t()  # (B, num_classes), each in [-1, 1]

        if labels is None:
            # Inference / eval: no margin, just scaled cosine similarity.
            return cosine * self.scale

        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(min=1e-9))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + margin)

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)

        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        output = one_hot * phi + (1.0 - one_hot) * cosine
        return output * self.scale


class SignLanguageArcFaceTCN(nn.Module):
    """
    Same encoder + TCN backbone as SignLanguageTCN, but replaces the
    plain softmax classifier with an embedding projection + ArcFace
    margin head -- suited for datasets with very many classes and very
    few (often 1) examples per class.
    During training, pass `labels` to forward() so the margin can be
    applied to the correct class. During eval/inference, call without
    labels to get plain (unmargined) scaled cosine similarities, which
    can be used for standard argmax classification OR for nearest-
    neighbor / prototype matching via `.embed()`.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 256,
        tcn_channels=(256, 256, 256, 256),
        kernel_size: int = 3,
        dropout: float = 0.2,
        embedding_dim: int = 256,
        arc_scale: float = 30.0,
        arc_margin: float = 0.3,
    ):
        super().__init__()
        self.encoder = LandmarkSequenceEncoder(input_dim, d_model, dropout=dropout)
        self.tcn = TemporalConvNet(
            d_model, list(tcn_channels), kernel_size=kernel_size, dropout=dropout
        )
        pooled_dim = tcn_channels[-1] * 2  # mean+max concat
        self.embedding_proj = nn.Sequential(
            nn.Linear(pooled_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        self.arc_head = ArcMarginProduct(
            embedding_dim, num_classes, scale=arc_scale, margin=arc_margin
        )

    def set_margin(self, margin: float):
        """Passthrough for margin warm-up during training."""
        self.arc_head.set_margin(margin)

    def embed(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """Returns L2-normalized embeddings (B, embedding_dim), no classifier head."""
        h = self.encoder(x)
        h = h.transpose(1, 2)
        h = self.tcn(h)
        pooled = masked_mean_max_pool(h, lengths)
        emb = self.embedding_proj(pooled)
        return F.normalize(emb, dim=1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None,
                labels: torch.Tensor = None) -> torch.Tensor:
        h = self.encoder(x)
        h = h.transpose(1, 2)
        h = self.tcn(h)
        pooled = masked_mean_max_pool(h, lengths)
        emb = self.embedding_proj(pooled)
        logits = self.arc_head(emb, labels)
        return logits

    @torch.no_grad()
    def predict(self, x: torch.Tensor, lengths: torch.Tensor = None):
        """Convenience method: returns (predicted_class_idx, softmax_probs)
        using plain (unmargined) cosine similarities -- standard ArcFace
        inference behavior."""
        self.eval()
        logits = self.forward(x, lengths, labels=None)
        probs = F.softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1)
        return pred, probs


class SignLanguageTCN(nn.Module):
    """
    Full pipeline model: encoder -> TCN -> classifier.
    Args:
        input_dim: dimensionality of the per-frame landmark feature
                   vector (see slr.landmarks.FEATURE_DIM).
        num_classes: number of sign classes (words) to predict.
        d_model: encoder embedding dimension / TCN input channels.
        tcn_channels: list of channel sizes, one per TCN layer, e.g.
                      [256, 256, 256, 256] for a 4-layer TCN.
        kernel_size: TCN convolution kernel size.
        dropout: dropout used throughout.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 256,
        tcn_channels=(256, 256, 256, 256),
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = LandmarkSequenceEncoder(input_dim, d_model, dropout=dropout)
        self.tcn = TemporalConvNet(
            d_model, list(tcn_channels), kernel_size=kernel_size, dropout=dropout
        )
        self.classifier = ClassifierHead(tcn_channels[-1], num_classes, dropout=dropout + 0.1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B, T, input_dim) landmark sequence, zero-padded to T.
        lengths: (B,) actual sequence lengths before padding (optional
                 but recommended so padding doesn't skew pooling).
        returns: (B, num_classes) logits
        """
        h = self.encoder(x)              # (B, T, d_model)
        h = h.transpose(1, 2)            # (B, d_model, T) for conv1d
        h = self.tcn(h)                  # (B, C_out, T)
        logits = self.classifier(h, lengths)
        return logits

    @torch.no_grad()
    def predict(self, x: torch.Tensor, lengths: torch.Tensor = None):
        """Convenience method: returns (predicted_class_idx, softmax_probs)."""
        self.eval()
        logits = self.forward(x, lengths)
        probs = F.softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1)
        return pred, probs
