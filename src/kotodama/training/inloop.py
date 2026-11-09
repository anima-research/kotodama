"""In-loop evaluations: small eval suites scored on the LIVE training weights.

Enabled by ``train.py --inloop_evals riders,induction[,setbind,uncheatable,nca_gates]``
(default off). At every eval point every rank scores its shard of each suite
in-process (no server, no checkpoint reload) on the raw module — unwrapped from
DDP and torch.compile, so no recompiles — under ``no_grad`` + ``eval()`` (bf16
autocast on CUDA), gathers to rank 0 in ONE ``all_gather_object``, and rank 0

* logs the summary metrics in one ``hm_log`` call (keys ``inloop/...``, e.g.
  ``inloop/riders/narrative/3shot/acc``, plus ``inloop/time_s``), and
* writes every per-row result to ``<checkpoint_dir>/inloop/step_<N>.jsonl``
  (first line = ``{"suite": "_meta", ...}`` with the metrics), so offline
  analysis can re-read or re-aggregate later.

Suites:

  riders       riders-lite (the binding benchmark's ``binding_bench.riders.lite``, from the
               ``bench`` extra): N items x 4 registers x {0shot, 3shot, 3shot_relet}, scored
               by the SAME function the server's ``/logprobs`` uses (``kotodama.serve.logprobs``)
  setbind      set binding (the binding benchmark's ``binding_bench.bindprobes.setbind``): every static
               item (no transfers) of a bindprobes items file, 3-shot, scored by the same function
               as ``/logprobs`` and read by the benchmark's ``set_metrics``. ``inloop/setbind/k1``
               is hit@1 on the one-object sets, the benchmark's static set-binding k1 read
  induction    random-token repeat loss drop (``kotodama.eval.induction``)
  uncheatable  bits/byte on the first docs of each domain (``kotodama.eval.uncheatable``)
  nca_gates    NCA ledger accuracy (``kotodama.eval.nca_gates``), for NCA-phase runs

Cadence (``--inloop_schedule``): piecewise token FRACTIONS of the run,
``"<start>:<interval>,..."`` with ``start`` a fraction or ``decay`` (the WSD
decay start). The default ``0:0.025,decay:0.005`` = every 2.5% of tokens, every
0.5% once the LR decay has begun; plus always the first and final steps. A point
fires at the step whose cumulative tokens first reach the threshold (the
``--save_schedule`` rule), so eval ``step_N`` and checkpoint ``step_N`` hold the
same weights. Resumes skip points whose file already exists.

DDP safety: the schedule is a pure function of the run geometry, so every rank
reaches the same eval points; each point issues a fixed number of collectives
(one all-reduce of the SIGTERM flag, one gather) whatever suites fail, and
suite errors are caught per rank and reported by rank 0 — never a deadlock.
A SIGTERM on any rank skips the eval point on all of them.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, TypeVar

import numpy as np
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

SUITES: tuple[str, ...] = ("riders", "setbind", "induction", "uncheatable", "nca_gates")
_SUITE_ALIASES = {"riders_lite": "riders", "uncheatable_mini": "uncheatable",
                  "nca": "nca_gates"}
DEFAULT_SUITES: tuple[str, ...] = ("riders",)
_DEFAULT_ALIASES = {"on", "default", "true", "1"}
DEFAULT_SCHEDULE = "0:0.025,decay:0.005"
# The set-binding read scores the static setbind items (no transfers) under the benchmark's 3-shot condition.
SETBIND_CONDITION = "3shot"
SETBIND_TOP_N = 5   # full-vocab top tokens kept per row, as the benchmark's grid runner keeps by default
INLOOP_SUBDIR = "inloop"
_FILE_RE = re.compile(r"^step_(\d+)\.jsonl$")

T = TypeVar("T")
LogFn = Callable[..., None]  # hm_log(step, **metrics)


# ── Spec parsing ──────────────────────────────────────────────────────────────

def parse_suites(spec: Optional[str]) -> list[str]:
    """``"riders,induction"`` -> ``["riders", "induction"]``; empty/None -> [];
    ``on``/``default`` -> the default set (riders-lite)."""
    if not spec:
        return []
    if spec.strip().lower() in _DEFAULT_ALIASES:
        return list(DEFAULT_SUITES)
    out: list[str] = []
    for raw in spec.split(","):
        name = raw.strip().lower().replace("-", "_")
        if not name:
            continue
        name = _SUITE_ALIASES.get(name, name)
        if name not in SUITES:
            raise ValueError(f"--inloop_evals: unknown suite {raw.strip()!r}; "
                             f"choose from {SUITES}")
        if name not in out:
            out.append(name)
    return out


def parse_schedule(spec: str) -> list[tuple[float | str, float]]:
    """``"0:0.025,decay:0.005"`` -> ``[(0.0, 0.025), ("decay", 0.005)]``."""
    phases: list[tuple[float | str, float]] = []
    for seg in (s.strip() for s in spec.split(",") if s.strip()):
        try:
            start_s, every_s = seg.split(":")
            every = float(every_s)
        except ValueError as e:
            raise ValueError(f"--inloop_schedule segment {seg!r} is not START:INTERVAL") from e
        if not 0.0 < every <= 1.0:
            raise ValueError(f"--inloop_schedule {seg!r}: interval must be in (0, 1]")
        start_s = start_s.strip().lower()
        if start_s == "decay":
            phases.append(("decay", every))
            continue
        try:
            start = float(start_s)
        except ValueError as e:
            raise ValueError(f"--inloop_schedule {seg!r}: start must be a fraction or 'decay'") from e
        if not 0.0 <= start < 1.0:
            raise ValueError(f"--inloop_schedule {seg!r}: start must be in [0, 1)")
        phases.append((start, every))
    if not phases:
        raise ValueError(f"--inloop_schedule {spec!r}: no phases")
    return phases


def inloop_schedule(total_steps: int, decay_start: int, spec: str = DEFAULT_SCHEDULE,
                    final_step: Optional[int] = None,
                    cum_tokens: Optional[Sequence[float]] = None) -> list[int]:
    """Sorted eval steps for a run of ``total_steps`` steps.

    ``decay_start``: first step of the LR decay (``muon.decay_start_step``).
    ``final_step``: last step that runs (default ``total_steps - 1``; smaller under
    ``--stop_at_step``) — always included, as is step 0; nothing past it.
    ``cum_tokens[s]``: tokens consumed once step ``s`` is done (only needed under a
    batch ramp; default = equal tokens per step). Fraction ``f`` fires at the
    first step whose cumulative tokens reach ``f`` of the total.
    """
    if total_steps < 1:
        return []
    last = total_steps - 1 if final_step is None else min(final_step, total_steps - 1)
    if last < 0:
        return []
    cum = (np.arange(1, total_steps + 1, dtype=np.float64) if cum_tokens is None
           else np.asarray(cum_tokens, dtype=np.float64))
    if cum.shape != (total_steps,):
        raise ValueError(f"cum_tokens has {cum.shape[0]} entries for {total_steps} steps")
    if np.any(np.diff(cum) < 0) or cum[-1] <= 0:
        raise ValueError("cum_tokens must be non-decreasing and positive")
    total = float(cum[-1])

    def start_frac(step: int) -> float:  # tokens consumed BEFORE `step`, as a fraction
        return 0.0 if step <= 0 else float(cum[min(step, total_steps) - 1]) / total

    phases = sorted(((start_frac(decay_start) if s == "decay" else float(s)), d)
                    for s, d in parse_schedule(spec))
    steps = {0, last}
    for k, (a, d) in enumerate(phases):
        b = phases[k + 1][0] if k + 1 < len(phases) else 1.0
        j = 0
        while True:
            f = a + j * d
            if f >= b - 1e-12 or f >= 1.0:
                break
            s = int(np.searchsorted(cum, f * total * (1 - 1e-12), side="left"))
            if s <= last:
                steps.add(s)
            j += 1
    return sorted(steps)


# ── Shard / merge (pure; the DDP contract) ────────────────────────────────────

def shard(seq: Sequence[T], rank: int, world_size: int) -> list[tuple[int, T]]:
    """This rank's slice ``seq[rank::world_size]``, tagged with global indices."""
    if not 0 <= rank < world_size:
        raise ValueError(f"rank {rank} outside world of {world_size}")
    return [(i, seq[i]) for i in range(rank, len(seq), world_size)]


def merge_shards(parts: Sequence[Sequence[tuple[int, T]]], n: int) -> list[T]:
    """Inverse of :func:`shard`: per-rank ``(index, value)`` lists -> one list of
    ``n`` in global order. Raises on a missing or duplicated index."""
    out: list[Any] = [None] * n
    seen = [False] * n
    for part in parts:
        for i, v in part:
            if not 0 <= i < n or seen[i]:
                raise ValueError(f"bad or duplicate shard index {i} (n={n})")
            out[i], seen[i] = v, True
    if not all(seen):
        raise ValueError(f"missing {seen.count(False)} of {n} shard results")
    return out


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class InloopConfig:
    suites: list[str]
    schedule: str = DEFAULT_SCHEDULE
    max_batch: int = 64                  # prompts per forward
    max_batch_tokens: int = 8192         # summed prompt tokens per forward
    riders_items: Optional[str] = None   # items.jsonl (None: generate the standard battery)
    riders_shots: Optional[str] = None   # shots.json (with riders_items)
    riders_n: int = 200                  # items per register
    riders_seed: int = 0
    riders_relet: bool = True
    riders_top_n: int = 5
    setbind_items: Optional[str] = None  # a bindprobes items.jsonl holding setbind items
    induction_n_seqs: int = 32
    induction_len: int = 128
    induction_seed: int = 0
    uncheatable_path: Optional[str] = None
    uncheatable_docs: int = 8            # leading docs per domain
    nca_gates_dir: Optional[str] = None
    nca_gates_max_items: int = 256       # leading items per ledger file

    @classmethod
    def from_args(cls, args: Any) -> "InloopConfig":
        g = lambda k, d: getattr(args, f"inloop_{k}", d)  # noqa: E731
        return cls(
            suites=parse_suites(g("evals", None)),
            schedule=g("schedule", DEFAULT_SCHEDULE) or DEFAULT_SCHEDULE,
            max_batch=g("max_batch", 64), max_batch_tokens=g("max_batch_tokens", 8192),
            riders_items=g("riders_items", None), riders_shots=g("riders_shots", None),
            riders_n=g("riders_n", 200), riders_seed=g("riders_seed", 0),
            riders_relet=g("riders_relet", True), riders_top_n=g("riders_top_n", 5),
            setbind_items=g("setbind_items", None),
            induction_n_seqs=g("induction_n_seqs", 32), induction_len=g("induction_len", 128),
            uncheatable_path=g("uncheatable_path", None),
            uncheatable_docs=g("uncheatable_docs", 8),
            nca_gates_dir=g("nca_gates_dir", None),
            nca_gates_max_items=g("nca_gates_max_items", 256),
        )


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """The raw module: out of DDP (``.module``) and torch.compile (``._orig_mod``)."""
    m = model.module if hasattr(model, "module") else model
    return getattr(m, "_orig_mod", m)


def _load_tokenizer() -> Any:
    from transformers import AutoTokenizer

    from kotodama.presets import TOKENIZER_NAME
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


# ── The evaluator ─────────────────────────────────────────────────────────────


def _bench(package: str, *modules: str) -> tuple[Any, ...]:
    """The binding benchmark's modules ``binding_bench.<package>.<name>``, or a refusal naming the extra."""
    from importlib import import_module

    try:
        return tuple(import_module(f"binding_bench.{package}.{m}") for m in modules)
    except ImportError as exc:
        raise RuntimeError("the in-loop riders and setbind suites need the binding benchmark: install "
                           "kotodama[bench], or put the binding-bench checkout on PYTHONPATH") from exc


def _bench_riders(*modules: str) -> tuple[Any, ...]:
    """The binding benchmark's riders modules (``binding_bench.riders.<name>``)."""
    return _bench("riders", *modules)


def _pinned_battery(path: Path) -> Optional[str]:
    """The benchmark's name for the bindprobes battery whose file pin ``path`` matches, after checking that its
    prompts render to that battery's render pin under the installed benchmark; None for an unpinned file.

    A bindprobes file stores item specs and its prompts are rendered at scoring time, so the file pin alone does
    not fix the strings; the render check refuses (``PinError``) a pinned file whose prompts the installed
    benchmark would render differently."""
    (bt,) = _bench("binding", "batteries")
    digest = bt.sha256(path)
    for b in bt.BATTERIES.values():
        if b.kind == "bindprobes" and b.sha256 == digest:
            bt.check_render(b, path)
            return str(b.name)
    return None


class InloopEvaluator:
    """Build once at train start (every rank, same args), call :meth:`plan`, then
    :meth:`run` at every step where :meth:`due` is true (every rank)."""

    def __init__(self, cfg: InloopConfig, *, out_dir: Path | str, rank: int,
                 world_size: int, device: torch.device, max_seq_len: int,
                 vocab_size: int, log_fn: Optional[LogFn] = None,
                 tokenizer: Any = None) -> None:
        if not cfg.suites:
            raise ValueError("InloopEvaluator needs at least one suite")
        self.cfg = cfg
        self.out_dir = Path(out_dir) / INLOOP_SUBDIR
        self.rank, self.world_size = rank, world_size
        self.device = torch.device(device)
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.log_fn = log_fn
        self.steps: list[int] = []
        self.pending: set[int] = set()
        self.notes: dict[str, Any] = {}
        need_tok = bool({"riders", "setbind", "uncheatable", "induction"} & set(cfg.suites))
        self.tok = tokenizer if tokenizer is not None else (_load_tokenizer() if need_tok else None)
        text_suites = sorted({"riders", "setbind", "uncheatable"} & set(cfg.suites))
        if text_suites and len(self.tok) > vocab_size:
            raise ValueError(f"in-loop {','.join(text_suites)} need the language vocab: "
                             f"tokenizer has {len(self.tok)} ids, model vocab is {vocab_size} "
                             f"(an NCA-phase run? use induction / nca_gates)")
        if "riders" in cfg.suites:
            self._prep_riders()
        if "setbind" in cfg.suites:
            self._prep_setbind()
        if "induction" in cfg.suites:
            self._prep_induction()
        if "uncheatable" in cfg.suites:
            self._prep_uncheatable()
        if "nca_gates" in cfg.suites:
            self._prep_nca_gates()

    # ── preparation (deterministic; identical on every rank) ─────────────────

    def _prep_riders(self) -> None:
        ri, lite = _bench_riders("items", "lite")
        cfg = self.cfg
        if (cfg.riders_items is None) != (cfg.riders_shots is None):
            raise ValueError("--inloop_riders_items and --inloop_riders_shots go together")
        if cfg.riders_items is not None:
            all_items = ri.load_items(Path(cfg.riders_items))
            shots = ri.load_items(Path(cfg.riders_shots))  # type: ignore[arg-type]
            source = cfg.riders_items
        else:  # the standard battery, byte-identical to `riders gen` defaults
            all_items, shots = ri.generate(list(ri.BANKED_VERIFIED_NAMES), 200, 200,
                                           ri.DEFAULT_SEED)
            source = f"generated(seed={ri.DEFAULT_SEED})"
        if len(shots) < ri.N_SHOTS:
            raise ValueError(f"riders shots: need {ri.N_SHOTS}, have {len(shots)}")
        shots = shots[:ri.N_SHOTS]
        sel = lite.select_items(all_items, cfg.riders_n, cfg.riders_seed)
        relet = lite.reletter_shots(shots, sel) if cfg.riders_relet else None
        conds = lite.CONDITIONS if cfg.riders_relet else ("0shot", "3shot")
        jobs = lite.build_jobs(sel, shots, relet, lite.REGISTERS, conds)
        lengths = [len(self.tok.encode(j.prompt)) for j in jobs]
        keep = [k for k, n in enumerate(lengths) if n < self.max_seq_len]
        if len(keep) < len(jobs):
            logger.warning("inloop riders: %d/%d prompts exceed max_seq_len %d — dropped",
                           len(jobs) - len(keep), len(jobs), self.max_seq_len)
        self.r_items = sel
        self.r_jobs = [jobs[k] for k in keep]
        self.r_lengths = [lengths[k] for k in keep]
        self.notes["riders"] = {"source": source, "n_items": len(sel),
                                "n_prompts": len(self.r_jobs),
                                "n_dropped_too_long": len(jobs) - len(keep),
                                "prompt_tokens": int(sum(self.r_lengths)),
                                "conditions": list(conds)}

    def _prep_setbind(self) -> None:
        bp_items, bp_grid = _bench("bindprobes", "items", "grid")
        if not self.cfg.setbind_items:
            raise ValueError("--inloop_evals setbind needs --inloop_setbind_items (a bindprobes items.jsonl)")
        path = Path(self.cfg.setbind_items)
        battery = _pinned_battery(path)
        if battery is None:
            logger.warning("inloop setbind: %s matches no pinned bindprobes battery; its values compare only "
                           "with grids scored on this same file", path)
        sel = [it for it in bp_items.load_items(path)
               if it.family == "setbind" and int(it.meta.get("n_transfers", -1)) == 0]
        if not sel:
            raise ValueError(f"inloop setbind: no static setbind items (family setbind, no transfers) in {path}")
        jobs = [bp_grid.build_payload_item(it, SETBIND_CONDITION) for it in sel]
        lengths = [len(self.tok.encode(j["prompt"])) for j in jobs]
        keep = [k for k, n in enumerate(lengths) if n < self.max_seq_len]
        if len(keep) < len(jobs):
            logger.warning("inloop setbind: %d/%d prompts exceed max_seq_len %d — dropped",
                           len(jobs) - len(keep), len(jobs), self.max_seq_len)
        self.s_items = [sel[k] for k in keep]
        self.s_jobs = [jobs[k] for k in keep]
        self.s_lengths = [lengths[k] for k in keep]
        ks: dict[str, int] = {}
        for it in self.s_items:
            ks[f"k{it.meta['k']}"] = ks.get(f"k{it.meta['k']}", 0) + 1
        self.notes["setbind"] = {"source": str(path), "battery": battery, "condition": SETBIND_CONDITION,
                                 "n_prompts": len(self.s_jobs), "n_by_k": dict(sorted(ks.items())),
                                 "n_dropped_too_long": len(jobs) - len(keep),
                                 "prompt_tokens": int(sum(self.s_lengths))}

    def _prep_induction(self) -> None:
        from kotodama.eval.induction import build_repeat_batch, candidate_ids
        L = self.cfg.induction_len
        special = sorted(set(getattr(self.tok, "all_special_ids", []) or []))
        exclude = [i for i in special if i < self.vocab_size] or [0]
        eos = getattr(self.tok, "eos_token_id", None)
        prepend = eos if eos is not None and eos < self.vocab_size else None
        offset = 0 if prepend is None else 1
        if offset + 2 * L > self.max_seq_len:
            raise ValueError(f"inloop induction: {offset + 2 * L} tokens > max_seq_len {self.max_seq_len}")
        self.i_ids = build_repeat_batch(self.cfg.induction_n_seqs, L, 0, self.vocab_size,
                                        exclude, seed=self.cfg.induction_seed,
                                        prepend_id=prepend)
        self.i_offset = offset
        self.i_ncand = len(candidate_ids(0, self.vocab_size, exclude))
        self.notes["induction"] = {"n_seqs": int(self.i_ids.shape[0]), "prefix_len": L,
                                   "prompt_tokens": int(self.i_ids.numel())}

    def _prep_uncheatable(self) -> None:
        from kotodama.eval.uncheatable import CHUNK_SIZE, load_domains
        if not self.cfg.uncheatable_path:
            raise ValueError("--inloop_evals uncheatable needs --inloop_uncheatable_path")
        if self.tok.bos_token_id is None:
            raise ValueError("inloop uncheatable: tokenizer has no BOS token")
        domains = load_domains(self.cfg.uncheatable_path)
        docs: list[tuple[str, int, list[int], int]] = []  # (domain, idx, ids, utf8 bytes)
        for name in sorted(domains):
            kept = 0
            for k, text in enumerate(domains[name]):
                if kept >= self.cfg.uncheatable_docs:
                    break
                ids = self.tok(text, add_special_tokens=False)["input_ids"]
                if len(ids) <= 1:
                    continue
                docs.append((name, k, ids, len(text.encode("utf-8"))))
                kept += 1
        if not docs:
            raise ValueError(f"inloop uncheatable: no usable docs under {self.cfg.uncheatable_path}")
        self.u_docs = docs
        self.u_bos = int(self.tok.bos_token_id)
        self.u_chunk = min(CHUNK_SIZE, self.max_seq_len - 1)
        self.notes["uncheatable"] = {"n_docs": len(docs), "n_domains": len({d[0] for d in docs}),
                                     "chunk_size": self.u_chunk,
                                     "prompt_tokens": int(sum(len(d[2]) for d in docs))}

    def _prep_nca_gates(self) -> None:
        from kotodama.eval.nca_gates import LEDGER_FILES, _read_jsonl, resolve_token_map
        if not self.cfg.nca_gates_dir:
            raise ValueError("--inloop_evals nca_gates needs --inloop_nca_gates_dir")
        ev = Path(self.cfg.nca_gates_dir)
        self.n_tm = resolve_token_map(ev)
        self.n_items: list[tuple[str, dict[str, Any]]] = []
        for name in LEDGER_FILES:
            p = ev / name
            if p.exists():
                rows = _read_jsonl(p)[: self.cfg.nca_gates_max_items]
                self.n_items += [(name, r) for r in rows]
        if not self.n_items:
            raise ValueError(f"inloop nca_gates: no eval_ledger*.jsonl under {ev}")
        self.notes["nca_gates"] = {"n_items": len(self.n_items),
                                   "prompt_tokens": int(sum(len(r["prefix"]) for _, r in self.n_items))}

    # ── schedule ──────────────────────────────────────────────────────────────

    def done_steps(self) -> set[int]:
        """Eval points already written (rank 0's view of the inloop dir)."""
        if not self.out_dir.is_dir():
            return set()
        return {int(m.group(1)) for p in self.out_dir.iterdir()
                if (m := _FILE_RE.match(p.name))}

    def plan(self, total_steps: int, decay_start: int, final_step: int, start_step: int = 0,
             cum_tokens: Optional[Sequence[float]] = None) -> list[int]:
        """Compute the schedule; drop points before ``start_step`` and already-written
        ones (rank 0 decides, broadcast). Collective under DDP. Returns pending steps."""
        self.steps = inloop_schedule(total_steps, decay_start, self.cfg.schedule,
                                     final_step=final_step, cum_tokens=cum_tokens)
        done = sorted(self.done_steps()) if self.rank == 0 else []
        if self._distributed():
            box: list[Any] = [done]
            dist.broadcast_object_list(box, src=0)
            done = box[0]
        self.pending = {s for s in self.steps if s >= start_step} - set(done)
        if self.rank == 0:
            logger.info("In-loop evals %s: %d points (%d pending; %d already written), "
                        "schedule %r, prep %s", ",".join(self.cfg.suites), len(self.steps),
                        len(self.pending), len(set(done) & set(self.steps)),
                        self.cfg.schedule, json.dumps(self.notes))
        return sorted(self.pending)

    def due(self, step: int) -> bool:
        return step in self.pending

    # ── one eval point ────────────────────────────────────────────────────────

    def _distributed(self) -> bool:
        return self.world_size > 1 and dist.is_available() and dist.is_initialized()

    def _any_stop(self, stop_requested: bool) -> bool:
        if not self._distributed():
            return stop_requested
        flag = torch.tensor([1.0 if stop_requested else 0.0], device=self.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item() > 0)

    def _gather(self, obj: Any) -> list[Any]:
        if not self._distributed():
            return [obj]
        out: list[Any] = [None] * self.world_size
        dist.all_gather_object(out, obj)
        return out

    def _model_fn(self, raw: torch.nn.Module) -> Callable[[torch.Tensor], dict[str, torch.Tensor]]:
        amp = (torch.autocast("cuda", dtype=torch.bfloat16) if self.device.type == "cuda"
               else nullcontext())

        def fn(ids: torch.Tensor) -> dict[str, torch.Tensor]:
            with amp:
                return raw(ids.to(self.device))
        return fn

    def _local(self, suite: str, model_fn: Callable[[torch.Tensor], dict[str, torch.Tensor]]) -> Any:
        r, w = self.rank, self.world_size
        if suite == "riders":
            from kotodama.serve.logprobs import score_logprob_items
            mine = shard(range(len(self.r_jobs)), r, w)
            jobs = [{"id": self.r_jobs[i].id, "prompt": self.r_jobs[i].prompt,
                     "candidates": self.r_jobs[i].candidates} for _, i in mine]
            res = score_logprob_items(
                lambda b: model_fn(b)["logits"], self.tok, jobs, self.cfg.riders_top_n,
                self.device, max_seq_len=self.max_seq_len, max_batch=self.cfg.max_batch,
                max_total_tokens=self.cfg.max_batch_tokens,
                lengths=[self.r_lengths[i] for _, i in mine])
            return [(i, x) for (i, _), x in zip(mine, res)]
        if suite == "setbind":
            from kotodama.serve.logprobs import score_logprob_items
            mine = shard(range(len(self.s_jobs)), r, w)
            res = score_logprob_items(
                lambda b: model_fn(b)["logits"], self.tok, [self.s_jobs[i] for _, i in mine],
                SETBIND_TOP_N, self.device, max_seq_len=self.max_seq_len,
                max_batch=self.cfg.max_batch, max_total_tokens=self.cfg.max_batch_tokens,
                lengths=[self.s_lengths[i] for _, i in mine])
            return [(i, x) for (i, _), x in zip(mine, res)]
        if suite == "induction":
            from kotodama.eval.induction import token_losses
            mine = shard(range(self.i_ids.shape[0]), r, w)
            if not mine:
                return []
            idx = [i for i, _ in mine]
            bs = max(1, self.cfg.max_batch_tokens // self.i_ids.shape[1])
            losses = token_losses(model_fn, self.i_ids[idx], batch_size=bs, device=str(self.device))
            return [(i, losses[k].tolist()) for k, i in enumerate(idx)]
        if suite == "uncheatable":
            from kotodama.eval.uncheatable import doc_nll
            out = []
            for i, (_, _, ids, _) in shard(self.u_docs, r, w):
                out.append((i, doc_nll(lambda x: model_fn(x)["logits"][0], ids, self.u_bos,
                                       self.u_chunk, "bf16", self.device)))
            return out
        if suite == "nca_gates":
            from kotodama.eval.nca_gates import score_ledger
            mine = shard(self.n_items, r, w)
            rows = score_ledger(model_fn, [it for _, (_, it) in mine], self.n_tm,
                                device=str(self.device))
            return [(i, row) for (i, _), row in zip(mine, rows)]
        raise ValueError(f"unknown suite {suite!r}")

    def run(self, step: int, model: torch.nn.Module, tokens_consumed: int = 0,
            stop_requested: bool = False) -> Optional[dict[str, float]]:
        """Score every suite at ``step``. EVERY rank must call this at the same steps.
        Returns rank 0's metrics (None elsewhere, or when skipped)."""
        self.pending.discard(step)
        if self._any_stop(stop_requested):
            if self.rank == 0:
                logger.info("inloop: step %d skipped (stop requested)", step)
            return None
        t0 = time.time()
        raw = unwrap_model(model)
        was_training = raw.training
        raw.eval()
        local: dict[str, dict[str, Any]] = {}
        try:
            model_fn = self._model_fn(raw)
            with torch.no_grad():
                for suite in self.cfg.suites:
                    ts = time.time()
                    try:
                        local[suite] = {"ok": True, "data": self._local(suite, model_fn)}
                    except Exception as e:  # report via rank 0; never desync the gather
                        logger.warning("inloop %s failed on rank %d at step %d: %s: %s",
                                       suite, self.rank, step, type(e).__name__, e)
                        local[suite] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                        if self.device.type == "cuda":
                            torch.cuda.empty_cache()
                    local[suite]["time_s"] = time.time() - ts
        finally:
            raw.train(was_training)
        parts = self._gather(local)
        if self.rank != 0:
            return None
        try:
            metrics, rows, errors = self._reduce(parts)
        except Exception as e:  # rank 0 only: telemetry must never stop the run
            logger.exception("inloop reduce failed at step %d: %s", step, e)
            return None
        metrics = {f"inloop/{k}": v for k, v in metrics.items()}
        metrics["inloop/time_s"] = time.time() - t0
        metrics["inloop/errors"] = float(len(errors))
        for suite in self.cfg.suites:
            metrics[f"inloop/{suite}/time_s"] = max(p[suite]["time_s"] for p in parts)
        self._write(step, tokens_consumed, metrics, rows, errors)
        if self.log_fn is not None:  # one batched call; NaN/inf are not valid JSON
            self.log_fn(step, **{k: v for k, v in metrics.items() if math.isfinite(v)})
        logger.info("inloop step %d: %.1fs, %d metrics%s", step, metrics["inloop/time_s"],
                    len(metrics), f", errors {errors}" if errors else "")
        return metrics

    # ── rank-0 reduction ──────────────────────────────────────────────────────

    def _reduce(self, parts: list[dict[str, dict[str, Any]]]
                ) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, str]]:
        metrics: dict[str, float] = {}
        rows: list[dict[str, Any]] = []
        errors: dict[str, str] = {}
        for suite in self.cfg.suites:
            bad = [f"rank{r}: {p[suite]['error']}" for r, p in enumerate(parts)
                   if not p[suite]["ok"]]
            if bad:
                errors[suite] = "; ".join(bad)
                continue
            data = [p[suite]["data"] for p in parts]
            try:
                m, rr = getattr(self, f"_reduce_{suite}")(data)
            except Exception as e:  # one suite's bug must not hide the others
                logger.exception("inloop %s reduce failed", suite)
                errors[suite] = f"reduce: {type(e).__name__}: {e}"
                continue
            metrics.update(m)
            rows += [{"suite": suite, **r} for r in rr]
        return metrics, rows, errors

    def _reduce_riders(self, data: list[Any]) -> tuple[dict[str, float], list[dict[str, Any]]]:
        lite, errors, grid = _bench_riders("lite", "errors", "grid")
        coarse_error_type, error_types, picked = errors.coarse_error_type, errors.error_types, errors.picked
        chat_family, score_row = grid.chat_family, grid.score_row
        results = merge_shards(data, len(self.r_jobs))
        chat_fmt = chat_family("koto")[1]
        rows = []
        for job, res in zip(self.r_jobs, results):
            item = self.r_items[job.item_index]
            row = score_row(item, res, job.renderer, job.nshots, "inloop",
                            chat_fmt if job.renderer == "chat" else None)
            row["condition"] = job.condition
            if row["gold_rank_cand"] != 1:
                pick = picked(row)
                row["pick"] = pick
                row["error_type"] = coarse_error_type(item, pick)
                row["error_types"] = error_types(item, pick)
            rows.append(row)
        bad = [r for r in rows if r["multi_token_candidates"]]
        if bad:
            raise ValueError(f"{len(bad)} rows scored multi-token candidates "
                             f"(first {bad[0]['item_id']}) — name verification violated")
        return lite.summarize(rows), rows

    def _reduce_setbind(self, data: list[Any]) -> tuple[dict[str, float], list[dict[str, Any]]]:
        bp_grid, bp_setbind = _bench("bindprobes", "grid", "setbind")
        results = merge_shards(data, len(self.s_jobs))
        rows = []
        for item, res in zip(self.s_items, results):
            row = bp_grid.score_row(item, res, SETBIND_CONDITION, "inloop")
            if row["multi_token_candidates"]:
                raise ValueError(f"{item.item_id} scored multi-token candidates "
                                 f"{row['multi_token_candidates']} — name verification violated")
            sm = bp_setbind.set_metrics(row["cand_lps"], row["gold_set"])
            row.update(hit1=sm["hit1"], exactk=sm["exactk"], set_margin=sm["margin"])
            rows.append(row)
        return setbind_summary(rows), rows

    def _reduce_induction(self, data: list[Any]) -> tuple[dict[str, float], list[dict[str, Any]]]:
        from kotodama.eval.induction import aggregate_induction
        n = int(self.i_ids.shape[0])
        losses = torch.tensor(merge_shards(data, n), dtype=torch.float32)
        agg = aggregate_induction(losses, self.cfg.induction_len, self.i_offset,
                                  n_candidates=self.i_ncand)
        keys = ("induction_gain", "induction_gain_normalized", "induction_gain_se",
                "loss_first", "loss_second")
        return ({f"induction/{k}": float(agg[k]) for k in keys}, [agg])

    def _reduce_uncheatable(self, data: list[Any]) -> tuple[dict[str, float], list[dict[str, Any]]]:
        from kotodama.eval.uncheatable import bpb_from_sums
        nll = merge_shards(data, len(self.u_docs))
        rows, per = [], {}
        for (dom, k, ids, nbytes), x in zip(self.u_docs, nll):
            rows.append({"domain": dom, "doc_index": k, "nll": x, "n_bytes": nbytes,
                         "n_tokens": len(ids)})
            per.setdefault(dom, ([], []))
            per[dom][0].append(x)
            per[dom][1].append(nbytes)
        m = {f"uncheatable/{d}/bpb": bpb_from_sums(a, b) for d, (a, b) in per.items()}
        m["uncheatable/mean_bpb"] = float(np.mean(list(m.values())))
        return m, rows

    def _reduce_nca_gates(self, data: list[Any]) -> tuple[dict[str, float], list[dict[str, Any]]]:
        from kotodama.eval.nca_gates import aggregate_ledger
        scored = merge_shards(data, len(self.n_items))
        by: dict[str, list[dict[str, Any]]] = {}
        rows = []
        for (fname, _), row in zip(self.n_items, scored):
            by.setdefault(fname, []).append(row)
            rows.append({"file": fname, **row})
        m: dict[str, float] = {}
        for fname, rr in by.items():
            tag = (fname.removeprefix("eval_ledger").removeprefix("_")
                   .removesuffix(".jsonl") or "legacy")
            agg = aggregate_ledger(rr)
            for split in ("all", "unique_write", "post_overwrite"):
                if agg.get(split):
                    m[f"nca_gates/{tag}/{split}/acc"] = agg[split]["acc"]
            if agg.get("all"):
                m[f"nca_gates/{tag}/mean_gold_lp"] = agg["all"]["mean_gold_lp"]
        return m, rows

    def _write(self, step: int, tokens: int, metrics: dict[str, float],
               rows: list[dict[str, Any]], errors: dict[str, str]) -> None:
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            path = self.out_dir / f"step_{step:08d}.jsonl"
            tmp = path.with_suffix(".jsonl.tmp")
            meta = {"suite": "_meta", "step": step, "tokens": tokens,
                    "world_size": self.world_size, "metrics": metrics, "errors": errors,
                    "config": asdict(self.cfg), "prep": self.notes}
            with tmp.open("w", encoding="utf-8") as fh:
                fh.write(json.dumps(meta, default=_json_default) + "\n")
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False, default=_json_default) + "\n")
            os.replace(tmp, path)
        except OSError as e:  # a full disk must not kill training
            logger.error("inloop: could not write step %d rows: %s", step, e)


def setbind_summary(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Set-binding metrics from scored rows (``k``, ``hit1``, ``exactk``, ``set_margin``), keys under ``setbind/``:

    ``k1``, ``k2``, ``k3``  hit@1 (the top candidate is held) on the sets of that size
    ``k23``                 hit@1 on the sets of size 2 and 3 together
    ``k23_exactk``          exact@k (the top k candidates are exactly the set) on sizes 2 and 3
    ``k1_margin``           mean margin, min logp(held) − max logp(not held), on the one-object sets
    ``k23_margin``          the same on sizes 2 and 3
    ``n_k1``, ``n_k23``     row counts

    Chance on these items (8 objects in play): hit@1 = k/8, exact@k = 1/C(8, k). A size absent from the rows
    leaves its keys out."""
    by: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        k = int(r["k"])
        by.setdefault(f"k{k}", []).append(r)
        if k > 1:
            by.setdefault("k23", []).append(r)
    m: dict[str, float] = {}
    for name in ("k1", "k2", "k3", "k23"):
        sub = by.get(name)
        if not sub:
            continue
        m[f"setbind/{name}"] = float(np.mean([r["hit1"] for r in sub]))
        if name in ("k1", "k23"):
            m[f"setbind/{name}_margin"] = float(np.mean([r["set_margin"] for r in sub]))
            m[f"setbind/n_{name}"] = float(len(sub))
        if name == "k23":
            m["setbind/k23_exactk"] = float(np.mean([r["exactk"] for r in sub]))
    return m


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, set):
        return list(o)
    raise TypeError(f"not JSON serializable: {type(o).__name__}")
