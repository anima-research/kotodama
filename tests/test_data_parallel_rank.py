"""Without TP each DDP rank must read its OWN data partition."""

from __future__ import annotations

import numpy as np

from kotodama.data.dataset import TokenizedDataset
from kotodama.training.train import data_parallel_rank


def test_without_tp_each_rank_is_its_own_partition():
    assert [data_parallel_rank(r, None) for r in range(8)] == list(range(8))


def test_partitions_differ_across_ranks(tmp_path):
    path = tmp_path / "toy.bin"
    np.arange(64 * 16, dtype=np.uint16).tofile(path)
    firsts = []
    for r in range(4):
        ds = TokenizedDataset(path=str(path), seq_len=16, rank=data_parallel_rank(r, None),
                              world_size=4, seed=42)
        firsts.append(next(iter(ds)))
    flat = [tuple(np.asarray(x).ravel().tolist()) for x in firsts]
    assert len(set(flat)) == 4   # four ranks, four different sequences
