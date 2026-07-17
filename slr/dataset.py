
"""
dataset.py
==========
Loads precomputed landmark sequences (.npy files, one per video, shape
(T, FEATURE_DIM)) from a directory laid out as:
    landmarks_root/
        WORD_A/
            0001.npy
            0002.npy
        WORD_B/
            0001.npy
        ...
and exposes them as a PyTorch Dataset that pads/truncates each sequence
to a fixed max length and returns (sequence, length, label_idx).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class LandmarkSequenceDataset(Dataset):
    def __init__(
        self,
        landmarks_root: str,
        max_len: int = 150,
        label_map: dict[str, int] | None = None,
        augment: bool = False,
    ):
        self.root = Path(landmarks_root)
        self.max_len = max_len
        self.augment = augment

        if label_map is None:
            words = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
            label_map = {w: i for i, w in enumerate(words)}
        self.label_map = label_map
        self.idx_to_label = {i: w for w, i in label_map.items()}

        self.samples: list[tuple[Path, int]] = []
        for word, idx in self.label_map.items():
            word_dir = self.root / word
            if not word_dir.is_dir():
                continue
            for npy_path in sorted(word_dir.glob("*.npy")):
                self.samples.append((npy_path, idx))

        if not self.samples:
            raise RuntimeError(f"No .npy samples found under {landmarks_root}")

    def __len__(self):
        return len(self.samples)

    def _load_and_fit(self, path: Path) -> tuple[np.ndarray, int]:
        seq = np.load(path).astype(np.float32)
        T = seq.shape[0]

        if T == 0:
            feat_dim = seq.shape[1] if seq.ndim == 2 else 1
            return np.zeros((self.max_len, feat_dim), dtype=np.float32), 0

        if T > self.max_len:
            idxs = np.linspace(0, T - 1, self.max_len).round().astype(int)
            seq = seq[idxs]
            length = self.max_len
        else:
            pad = np.zeros((self.max_len - T, seq.shape[1]), dtype=np.float32)
            seq = np.concatenate([seq, pad], axis=0)
            length = T

        if self.augment:
            noise = np.random.normal(0, 0.01, size=seq.shape).astype(np.float32)
            seq = seq + noise

        return seq, length

    def __getitem__(self, i: int):
        path, label = self.samples[i]
        seq, length = self._load_and_fit(path)
        return (
            torch.from_numpy(seq),
            torch.tensor(length, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )

    def save_label_map(self, out_path: str):
        with open(out_path, "w") as f:
            json.dump(self.label_map, f, indent=2)

    @staticmethod
    def load_label_map(path: str) -> dict[str, int]:
        with open(path) as f:
            return json.load(f)


def collate_fn(batch):
    seqs, lengths, labels = zip(*batch)
    return torch.stack(seqs), torch.stack(lengths), torch.stack(labels)
