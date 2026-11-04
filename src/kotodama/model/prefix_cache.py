"""Multi-entry prefix KV cache for the DecodeEngine: many recent prompts' KV in a bounded GPU pool.

The engine's own prefix cache (``DecodeEngine.prefill_cached``) reuses the previous request's KV only. Under
cross-request batching many conversations interleave on one engine, so the previous request rarely shares a prefix
with the next one, although almost every request shares a long prefix with SOME recent request (a tree search
expands siblings whose prompts differ only in their last turn or two). This module keeps the KV of recent prompts in
fixed-size token blocks and, for a new prompt, loads its longest cached prefix into the engine's KV cache so only
the suffix is prefilled.

Two parts:
  * ``BlockIndex`` (pure Python): a radix tree over token blocks. A node is one full block of ``block_size`` tokens
    whose key is (parent node, the block's tokens), so a node stands for its whole prefix: two prompts reach the
    same node exactly when their first ``k * block_size`` tokens are equal. Each node owns one slot of the pool.
    ``match`` walks a prompt's blocks down the tree; ``insert`` adds the blocks the walk did not find, evicting the
    least recently used leaves when the pool is full (a node is evicted only after all its children, so every
    cached node's ancestors are cached and a match is always a contiguous prefix).
  * ``PrefixKVCache`` (GPU): the pool tensors (every layer's K and V for ``n_blocks * block_size`` token positions,
    laid out like the engine's cache so a block run gathers into the engine's layout with one index op per tensor)
    and ``prefill``, a drop-in for ``DecodeEngine.prefill_cached`` that first loads the longest pooled prefix the
    engine does not already hold and afterwards stores the prompt's new full blocks.

Only KV written by a prefill (a prompt position, never a decode step) enters the pool: decode steps inject steering,
and the pool is keyed by tokens alone, so a steered position's KV must never be served to another request
(``DecodeEngine.prompt_kv_len`` marks where prefill-written KV ends). Stored blocks are checked finite: a non-finite
block is dropped instead of pooled, so it cannot poison later requests.

Memory: ``n_blocks * block_size`` positions of every layer's K and V, i.e. ``2 * n_layers * n_kv_heads * head_dim *
dtype bytes`` per token (the same per-token cost as one row of the engine's static cache). Loading is a gather plus
one copy per layer tensor; storing is a stack plus one indexed copy. Both are memory-bound copies of the block run,
far cheaper than prefilling it.
"""

from __future__ import annotations

import heapq
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch

from kotodama.model.decode_engine import DecodeEngine

logger = logging.getLogger(__name__)

_ROOT = -1


@dataclass
class _Node:
    slot: int
    parent: int  # parent slot, _ROOT for a first block
    key: tuple[int, ...]
    children: int = 0
    last_used: int = 0


@dataclass
class BlockIndex:
    """Radix index of token blocks over ``n_blocks`` pool slots (see the module docstring).

    ``match(ids)`` returns the slots of the longest cached run of ``ids``'s full blocks, in order, and marks them
    used. ``insert(ids, n_full)`` makes the first ``n_full`` blocks of ``ids`` cached and returns the (block index,
    slot) pairs it newly allocated: the caller fills those slots. It never evicts a block of ``ids``'s own chain, so
    when the pool is smaller than the chain the tail is left uncached. ``discard(slots)`` drops freshly inserted
    slots again (deepest first), for a store that failed."""

    n_blocks: int
    block_size: int
    nodes: dict[int, _Node] = field(default_factory=dict)
    children: dict[tuple[int, tuple[int, ...]], int] = field(default_factory=dict)
    free: list[int] = field(default_factory=list)
    clock: int = 0
    evicted: int = 0

    def __post_init__(self) -> None:
        if self.n_blocks < 1 or self.block_size < 1:
            raise ValueError(f"need n_blocks >= 1 and block_size >= 1, got {self.n_blocks}, {self.block_size}")
        self.free = list(range(self.n_blocks - 1, -1, -1))  # pop() hands out slot 0 first

    def __len__(self) -> int:
        return len(self.nodes)

    def _block(self, ids: Sequence[int], i: int) -> tuple[int, ...]:
        b = self.block_size
        return tuple(int(t) for t in ids[i * b:(i + 1) * b])

    def match(self, ids: Sequence[int], max_blocks: Optional[int] = None) -> list[int]:
        n = len(ids) // self.block_size
        if max_blocks is not None:
            n = min(n, max_blocks)
        self.clock += 1
        out: list[int] = []
        parent = _ROOT
        for i in range(n):
            slot = self.children.get((parent, self._block(ids, i)))
            if slot is None:
                break
            self.nodes[slot].last_used = self.clock
            out.append(slot)
            parent = slot
        return out

    def insert(self, ids: Sequence[int], n_full: int) -> list[tuple[int, int]]:
        n_full = min(n_full, len(ids) // self.block_size)
        self.clock += 1
        chain: list[int] = []
        parent = _ROOT
        i = 0
        while i < n_full:
            slot = self.children.get((parent, self._block(ids, i)))
            if slot is None:
                break
            self.nodes[slot].last_used = self.clock
            chain.append(slot)
            parent = slot
            i += 1
        need = n_full - i
        if need > len(self.free):
            self._evict(need - len(self.free), protect=set(chain))
        new: list[tuple[int, int]] = []
        while i < n_full and self.free:
            slot = self.free.pop()
            key = self._block(ids, i)
            self.nodes[slot] = _Node(slot, parent, key, last_used=self.clock)
            self.children[(parent, key)] = slot
            if parent != _ROOT:
                self.nodes[parent].children += 1
            new.append((i, slot))
            parent = slot
            i += 1
        return new

    def discard(self, slots: Sequence[int]) -> None:
        for slot in sorted(slots, key=self._depth, reverse=True):
            node = self.nodes.get(slot)
            if node is None:
                continue
            if node.children:
                raise RuntimeError(f"slot {slot} still has {node.children} cached children")
            self._remove(node)

    def _depth(self, slot: int) -> int:
        d = 0
        node = self.nodes.get(slot)
        while node is not None and node.parent != _ROOT:
            d += 1
            node = self.nodes.get(node.parent)
        return d

    def _remove(self, node: _Node) -> None:
        del self.nodes[node.slot]
        del self.children[(node.parent, node.key)]
        if node.parent != _ROOT:
            self.nodes[node.parent].children -= 1
        self.free.append(node.slot)

    def _evict(self, k: int, protect: set[int]) -> None:
        """Free up to ``k`` slots, least recently used leaf first; a parent becomes a candidate once its last child
        goes. Protected slots (the chain being extended) and their ancestors are never evicted."""
        heap = [(n.last_used, n.slot) for n in self.nodes.values() if n.children == 0 and n.slot not in protect]
        heapq.heapify(heap)
        while k > 0 and heap:
            _, slot = heapq.heappop(heap)
            node = self.nodes[slot]
            self._remove(node)
            self.evicted += 1
            k -= 1
            if node.parent != _ROOT:
                par = self.nodes[node.parent]
                if par.children == 0 and par.slot not in protect:
                    heapq.heappush(heap, (par.last_used, par.slot))

    def clear(self) -> None:
        self.nodes.clear()
        self.children.clear()
        self.free = list(range(self.n_blocks - 1, -1, -1))


class PrefixKVCache:
    """A bounded pool of prompt KV blocks in front of one DecodeEngine (see the module docstring).

    ``prefill(input_ids)`` has ``DecodeEngine.prefill_cached``'s contract (the engine holds the prompt's KV and
    position afterwards; returns last-position logits (1, V) fp32 and an info dict) and adds to the info dict
    ``pool_tokens`` (prompt tokens whose KV came from the pool) and ``stored_blocks``. ``stats`` accumulates
    requests, hits, prompt/reused/pooled tokens, stored/evicted blocks and host time spent matching and loading,
    in the engine's prefill or extend, and storing (host time: the GPU work is waited on only where a step syncs).
    """

    def __init__(self, engine: DecodeEngine, n_tokens: int, block_size: int = 16) -> None:
        if block_size < engine.min_cached_prefix:
            raise ValueError(f"block_size {block_size} is below the engine's min_cached_prefix "
                             f"{engine.min_cached_prefix}: a one-block hit would never be used")
        n_blocks = int(n_tokens) // int(block_size)
        if n_blocks < 1:
            raise ValueError(f"n_tokens {n_tokens} holds no {block_size}-token block")
        self.engine = engine
        self.block_size = int(block_size)
        self.index = BlockIndex(n_blocks, self.block_size)
        shape = (engine.n_layers, engine.n_kv_heads, n_blocks * self.block_size, engine.head_dim)
        self.k = torch.zeros(shape, device=engine.device, dtype=engine.dtype)
        self.v = torch.zeros(shape, device=engine.device, dtype=engine.dtype)
        self._offsets = torch.arange(self.block_size, device=engine.device, dtype=torch.long)
        self.stats: dict[str, float] = {"requests": 0, "hits": 0, "prompt_tokens": 0, "reused_tokens": 0,
                                        "pool_tokens": 0, "stored_blocks": 0, "evicted_blocks": 0,
                                        "dropped_blocks": 0, "load_s": 0.0, "extend_s": 0.0, "store_s": 0.0}
        logger.info("PrefixKVCache: %d blocks x %d tokens, %.1f GB", n_blocks, self.block_size,
                    2 * self.k.numel() * self.k.element_size() / 1e9)

    @property
    def n_blocks(self) -> int:
        return self.index.n_blocks

    def _positions(self, slots: Sequence[int]) -> torch.Tensor:
        """Pool token positions of ``slots``' blocks, in order: (len(slots) * block_size,)."""
        s = torch.tensor(list(slots), dtype=torch.long).to(self.engine.device, non_blocking=True)
        return (s.view(-1, 1) * self.block_size + self._offsets).view(-1)

    @torch.inference_mode()
    def _load(self, slots: Sequence[int], first_block: int) -> None:
        """Copy pooled blocks ``slots[first_block:]`` into the engine's cache at their prompt positions."""
        eng = self.engine
        lo, hi = first_block * self.block_size, len(slots) * self.block_size
        pos = self._positions(slots[first_block:])
        k = self.k.index_select(2, pos)  # (L, kv, hi - lo, hd)
        v = self.v.index_select(2, pos)
        dst_k = [c[0, :, lo:hi] for c in eng.k_caches]
        dst_v = [c[0, :, lo:hi] for c in eng.v_caches]
        torch._foreach_copy_(dst_k + dst_v, list(k.unbind(0)) + list(v.unbind(0)))

    @torch.inference_mode()
    def _store(self, new: list[tuple[int, int]]) -> int:
        """Copy the engine's KV of the new blocks into their slots; drop them all when any is non-finite. Returns
        the number of blocks stored."""
        eng = self.engine
        b = self.block_size
        first = new[0][0]
        # new blocks are consecutive prompt blocks (insert extends one chain)
        lo, hi = first * b, (new[-1][0] + 1) * b
        k = torch.stack([c[0, :, lo:hi] for c in eng.k_caches])  # (L, kv, hi - lo, hd)
        v = torch.stack([c[0, :, lo:hi] for c in eng.v_caches])
        if not bool((k.isfinite().all() & v.isfinite().all()).item()):
            logger.warning("PrefixKVCache: non-finite KV in prompt positions [%d, %d); not pooled", lo, hi)
            self.index.discard([s for _, s in new])
            self.stats["dropped_blocks"] += len(new)
            return 0
        pos = self._positions([s for _, s in new])
        self.k.index_copy_(2, pos, k)
        self.v.index_copy_(2, pos, v)
        return len(new)

    @torch.inference_mode()
    def prefill(self, input_ids: torch.Tensor, ids: Optional[Sequence[int]] = None,
                prompt_kv_only: bool = False) -> tuple[torch.Tensor, dict]:
        """``DecodeEngine.prefill_cached`` with the pool in front of it. ``ids``: the prompt's token ids as a host
        list when the caller has them (saves a device read of ``input_ids``). ``prompt_kv_only``: as on
        ``prefill_cached`` (the pool itself only ever holds prompt KV)."""
        eng = self.engine
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(f"Expected (1, S) input_ids, got {tuple(input_ids.shape)}")
        plen = input_ids.shape[1]
        if plen >= eng.max_seq_len:
            return eng.prefill_cached(input_ids)  # refuses with the engine's own error and state reset
        if ids is None:
            ids = input_ids[0].tolist()
        b = self.block_size
        t0 = time.perf_counter()
        # Cap at plen - 1 like prefill_cached: the last prompt token is always forwarded for next-token logits.
        slots = self.index.match(ids, max_blocks=(plen - 1) // b)
        pooled = len(slots) * b
        own = eng.cached_prefix_len(input_ids)
        loaded = 0
        if pooled > own and pooled >= eng.min_cached_prefix:
            # The engine keeps what it already holds of this prompt, as long as prefill wrote it (KV from decode
            # steps may carry steering); the pool supplies the rest.
            first_block = min(own, eng.prompt_kv_len) // b
            self._load(slots, first_block)
            eng.adopt_prefix(input_ids, pooled)
            loaded = pooled - first_block * b
        t1 = time.perf_counter()
        logits, info = eng.prefill_cached(input_ids, prompt_kv_only=prompt_kv_only)
        t2 = time.perf_counter()
        stored = 0
        n_full = min(plen, eng.prompt_kv_len) // b
        if n_full > len(slots):
            evicted0 = self.index.evicted
            new = self.index.insert(ids, n_full)
            self.stats["evicted_blocks"] += self.index.evicted - evicted0
            if new:
                stored = self._store(new)
        t3 = time.perf_counter()
        st = self.stats
        st["requests"] += 1
        st["hits"] += int(loaded > 0)
        st["prompt_tokens"] += plen
        st["reused_tokens"] += info.get("common_prefix", 0)
        st["pool_tokens"] += loaded
        st["stored_blocks"] += stored
        st["load_s"] += t1 - t0
        st["extend_s"] += t2 - t1
        st["store_s"] += t3 - t2
        info = {**info, "pool_tokens": loaded, "stored_blocks": stored}
        return logits, info

    def clear(self) -> None:
        """Forget every pooled block (the pool memory stays allocated)."""
        self.index.clear()
