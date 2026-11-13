"""Two micro-batches on two CUDA streams: does running one micro-batch's bandwidth-bound kernels (norms, routing,
residuals) under the other's matmuls beat batching them?

At the narrow-deep shapes the GPU is ~95% busy and a quarter or more of that busy time is bandwidth-bound work that
leaves the tensor cores idle. Two independent micro-batches launched on two streams can overlap on the GPU (the host
runs ahead, so both streams have work queued); one joint backward then walks both graphs, each op's backward on the
stream its forward ran on. Same math as two sequential micro-batches (gradients sum; the order of the two
accumulations may differ).

Modes, all priced per 2 sequences of --seq tokens (fwd+bwd, no optimizer):
  mb2      one micro-batch of 2 sequences (the ordinary way: bigger matmuls)
  seq2x1   two micro-batches of 1, one after the other on one stream
  par2x1   two micro-batches of 1 on two streams, joint backward

  tools/run_py.sh scripts/benchmark/stream_overlap.py --hidden 3072 --heads 24 --kv 12 --ffn 8192 --layers 48 \\
      --n_blocks 8 [--regional]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from kotodama.model.llama import KotodamaModel, KotodamaModelConfig  # noqa: E402

DEVICE = "cuda"
MODES = ("mb2", "seq2x1", "par2x1")


def _inputs(B: int, T: int, vocab: int, seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    ids = torch.randint(0, vocab, (B, T), device=DEVICE, generator=g)
    return {"ids": ids, "position_ids": torch.arange(T, device=DEVICE).expand(B, T),
            "cu_seqlens": (torch.arange(B + 1, device=DEVICE) * T).to(torch.int32)}


def _loss(fn: Callable, x: dict[str, torch.Tensor], T: int) -> torch.Tensor:
    logits = fn(x["ids"], position_ids=x["position_ids"], cu_seqlens=x["cu_seqlens"], max_seqlen=T)["logits"]
    return torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                                             x["ids"][:, 1:].reshape(-1))


def _time(step: Callable[[], None], iters: int) -> float:
    for _ in range(3):
        step()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        step()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hidden", type=int, required=True)
    ap.add_argument("--heads", type=int, required=True)
    ap.add_argument("--kv", type=int, required=True)
    ap.add_argument("--head_dim", type=int, default=128)
    ap.add_argument("--ffn", type=int, required=True)
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--n_blocks", type=int, default=8)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    ap.add_argument("--regional", action="store_true",
                    help="compile per sublayer (routing eager) instead of the whole model")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if not torch.cuda.is_available():
        print("needs CUDA", file=sys.stderr)
        return 2

    cfg = KotodamaModelConfig(
        hidden_size=a.hidden, num_layers=a.layers, num_attention_heads=a.heads, num_kv_heads=a.kv,
        head_dim=a.head_dim, intermediate_size=a.ffn, max_position_embeddings=a.seq, qk_norm=True,
        use_liger=True, rope_fp32=True, attn_impl="fa4", attn_res=a.n_blocks > 0, attn_res_n_blocks=max(a.n_blocks, 1),
    )
    torch.manual_seed(0)
    model = KotodamaModel(cfg).to(device=DEVICE, dtype=torch.bfloat16).train()
    if a.regional:
        model.compile_regions()
        fn: Callable = model
    else:
        fn = torch.compile(model)
    T = a.seq
    both = _inputs(2, T, cfg.vocab_size, 0)
    one = [_inputs(1, T, cfg.vocab_size, 1), _inputs(1, T, cfg.vocab_size, 2)]
    side = torch.cuda.Stream()

    def mb2() -> None:
        _loss(fn, both, T).backward()

    def seq2x1() -> None:
        for x in one:
            _loss(fn, x, T).backward()

    def par2x1() -> None:
        main_stream = torch.cuda.current_stream()
        side.wait_stream(main_stream)
        la = _loss(fn, one[0], T)
        with torch.cuda.stream(side):
            lb = _loss(fn, one[1], T)
        main_stream.wait_stream(side)
        (la + lb).backward()

    steps = {"mb2": mb2, "seq2x1": seq2x1, "par2x1": par2x1}
    out: dict[str, object] = {"tag": a.tag, "hidden": a.hidden, "ffn": a.ffn, "layers": a.layers,
                              "n_blocks": a.n_blocks, "seq": T, "regional": a.regional}
    for mode in a.modes:
        try:
            out[f"{mode}_ms"] = _time(steps[mode], a.iters)
        except Exception as e:  # noqa: BLE001 — report the mode's failure, keep the others
            out[f"{mode}_error"] = f"{type(e).__name__}: {e}"[:300]
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    if "mb2_ms" in out and "par2x1_ms" in out:
        out["par_vs_mb2"] = out["mb2_ms"] / out["par2x1_ms"]
    if "seq2x1_ms" in out and "par2x1_ms" in out:
        out["par_vs_seq"] = out["seq2x1_ms"] / out["par2x1_ms"]
    out["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
    print(json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
