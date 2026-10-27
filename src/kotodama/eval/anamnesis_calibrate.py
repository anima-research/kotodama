"""Calibrate a kotodama model for anamnesis: its positional means and residual basis.

anamnesis's ``run_calibration`` loads a Hugging Face checkpoint, which a kotodama checkpoint is
not (an HF cast would drop the AttnRes routing). This is the same calibration with only the two
model-facing steps swapped for kotodama's:

* **sampling** each ruler prompt with :func:`kotodama.model.capture.generate_light`, seeded by
  the prompt's index, under the decode policy the model's anamnesis registry row names;
* **reading states** by replaying each sequence once through the instrumented forward, in the
  hidden-state convention :func:`kotodama.model.capture.replay_extract_koto` reads features
  in (:func:`kotodama.model.capture.hidden_rows`: the embedding, then each layer's output).

Everything else is anamnesis's own, imported rather than restated: the pinned ruler prompts,
the decode policy, the fit, the files and the build receipt. Its refusals carry over: existing
positional means are reused unless ``--refit_means`` is given, and an existing basis is never
overwritten without ``--refit_basis``.

``--append_from BASE`` extends a calibration instead of fitting one
(:mod:`kotodama.eval.calibration_append`): a NEW directory holding BASE's basis byte for byte,
BASE's means and counts verbatim at every position BASE fills, and a tail fit from the ruler
prompts sampled to ``--max_new_tokens``, each still seeded by its index. The tail is fit only
after BASE's own token streams, replayed and reduced the same way, rebuild BASE's means byte
for byte. ``--shard I/N`` samples one shard's streams and stops; a run without it samples
whatever is missing, then fits. Where the installed anamnesis can verify an extension, its
record is written, so the extended directory keeps BASE's identity.

The registry row comes from :mod:`kotodama.anamnesis_registry`, whose file is added to
``ANAMNESIS_MODELS`` before anamnesis reads its registry. The forward runs in float32 with
deterministic algorithms and TF32 off, the arithmetic of kotodama's capture lane.

    tools/run_py.sh -m kotodama.eval.anamnesis_calibrate --checkpoint CKPT --out_dir DIR
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from collections.abc import Iterator
from pathlib import Path

from kotodama import anamnesis_registry, presets
from kotodama.eval import calibration_append

logger = logging.getLogger(__name__)


def _register() -> None:
    """Name kotodama's registry file in ANAMNESIS_MODELS (once), before anamnesis reads it."""
    current = [p for p in os.environ.get("ANAMNESIS_MODELS", "").split(os.pathsep) if p]
    path = str(anamnesis_registry.REGISTRY_FILE)
    if path not in current:
        os.environ["ANAMNESIS_MODELS"] = os.pathsep.join([*current, path])


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="kotodama-tsumugi-1b", choices=sorted(anamnesis_registry.MODELS))
    p.add_argument("--checkpoint", required=True, help="A .pt/.pt.zst path or a kotodama.ckpt name")
    p.add_argument("--out_dir", type=Path, required=True, help="The calibration directory to write")
    p.add_argument("--num_prompts", type=int, default=None, help="Use only the first N ruler prompts")
    p.add_argument("--max_new_tokens", type=int, default=None, help="Default: the registry row's budget")
    p.add_argument("--suppress_eos", action="store_true",
                   help="Generate past the stop tokens to the full budget")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--refit_means", action="store_true",
                   help="Recompute positional means that already exist (invalidates signatures made against them)")
    p.add_argument("--refit_basis", action="store_true", help="Replace an existing basis")
    p.add_argument("--dry_run", action="store_true", help="Print the configuration and stop")
    p.add_argument("--append_from", type=Path, default=None,
                   help="Extend the calibration in this directory into --out_dir instead of fitting one")
    p.add_argument("--shard", default=None, help="With --append_from: sample shard I/N only, then stop")
    p.add_argument("--count_floor", type=int, default=calibration_append.DEFAULT_TAIL_COUNT_FLOOR,
                   help="With --append_from: a tail position at or under this many streams stays zero")
    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parser().parse_args(argv)
    _register()

    from anamnesis.config import ExtractionConfig, resolve_preset
    from anamnesis.extraction import calibration_fit as cf
    from anamnesis.extraction.calibration import PCA_MODEL_NAME, POSITIONAL_MEANS_NAME

    key = args.model
    shape = anamnesis_registry.MODELS[key]["shape"]
    preset = resolve_preset(key)
    try:
        ruler = cf.calibration_prompts()
    except cf.CalibrationFitError as exc:
        raise SystemExit(str(exc)) from exc
    prompts = ruler if args.num_prompts is None else ruler[: args.num_prompts]
    settings = cf.generation_settings(preset, max_new_tokens=args.max_new_tokens)
    width = settings.max_new_tokens + cf.PROMPT_HEADROOM
    n_components = ExtractionConfig.from_preset(preset).pca_components
    means_path = args.out_dir / POSITIONAL_MEANS_NAME
    basis_path = args.out_dir / PCA_MODEL_NAME

    print(f"model {key} (kotodama shape {shape}) from {args.checkpoint}")
    print(f"  prompts {len(prompts)} bare (base model)   decode: temperature {settings.temperature} "
          f"top_p {settings.top_p} tokens {settings.max_new_tokens}   stop tokens "
          f"{'suppressed' if args.suppress_eos else list(settings.eos_token_ids)}")
    print(f"  means table {width} positions   basis per layer, corrected, {n_components} components")
    print(f"  -> {means_path}  {basis_path}")
    if args.dry_run:
        return
    if args.append_from is not None:
        _append(args, preset=preset, shape=shape, prompts=prompts, settings=settings, cf=cf)
        return
    if args.shard is not None:
        raise SystemExit("--shard samples an append's streams; it needs --append_from")
    if basis_path.exists() and not args.refit_basis:
        raise SystemExit(f"a basis is already at {basis_path}; pass --refit_basis to replace it")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    existing_means = cf.read_existing_means(means_path, args.refit_means)

    import torch

    from kotodama.model.capture import cached_forward, generate_light, hidden_rows

    model, tokenizer, lane, meta = _load(args, key, shape, preset)
    num_layers = int(model.config.num_layers)

    # 1. sample, nothing captured
    entries = {}
    started = time.perf_counter()
    eos = None if args.suppress_eos else list(settings.eos_token_ids)
    top_p = settings.top_p if settings.top_p < 1.0 else None
    with torch.no_grad():
        for index, text in enumerate(prompts):
            prompt = cf.encode_prompt(tokenizer, text, chat_template=False)
            torch.manual_seed(index)
            out = generate_light(model, torch.tensor([prompt], device=args.device), settings.max_new_tokens,
                                 do_sample=settings.do_sample, temperature=settings.temperature, top_p=top_p,
                                 eos_token_id=eos, output_hidden_states=False)
            entries[index] = cf.entry_from_ids(out.sequences[0].tolist(), len(prompt))
            elapsed = time.perf_counter() - started
            logger.info(f"calibration tokens {index + 1}/{len(prompts)}: {entries[index].n_gen} generated, "
                        f"{elapsed:.0f}s elapsed, about {elapsed / (index + 1) * (len(prompts) - index - 1):.0f}s left")
    tokens = cf.manifest_from_entries(entries)

    # 2. replay each sequence once for its states, in the koto hidden-state convention
    def states() -> Iterator[cf.PromptStates]:
        for gen_id in tokens.gen_ids():
            entry = tokens.entry(gen_id)
            ids = torch.tensor([entry.input_ids[:-1]], device=args.device)
            with torch.no_grad():
                _, cap, _ = cached_forward(model, ids, past=None, position_offset=0, light=False,
                                           capture_hidden=True)
                stacked = torch.stack([h[0] for h in hidden_rows(cap, num_layers)]).float().cpu().numpy()
            del cap
            length = entry.prompt_length
            yield cf.PromptStates(prompt_length=length, prefill=stacked[:, :length],
                                  steps=tuple(stacked[:, p] for p in range(length, stacked.shape[1])))

    try:
        fit = cf.fit_calibration(states(), preset=preset, settings=settings, n_components=n_components,
                                 pooled=False, existing_means=existing_means, max_positions=width)
        cf.write_calibration(fit, means_path, basis_path, required_through=None, tokens=tokens)
    except cf.CalibrationFitError as exc:
        raise SystemExit(str(exc)) from exc
    receipt = cf.build_receipt(
        fit, model=key, model_id=f"{key} {args.checkpoint} step {meta.get('step')}", prompts=prompts,
        settings=settings, suppress_eos=args.suppress_eos, chat_template=False, pooled=False,
        n_components=n_components, means_path=means_path, basis_path=basis_path, required_through=None,
        tokens_written=True)
    cf.write_build_receipt(receipt, basis_path)
    logger.info(f"calibration complete: {means_path}, {basis_path}; lane {lane}")



def _load(args: argparse.Namespace, key: str, shape: str, preset):
    """(model, tokenizer, lane, checkpoint meta): the checkpoint in float32 under the capture
    lane's arithmetic (deterministic algorithms, TF32 off)."""
    import torch
    from transformers import AutoTokenizer

    from kotodama.eval.model_loader import load_model
    from kotodama.model.capture import assert_lane_pinned

    lane = assert_lane_pinned()
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    meta: dict = {}
    model = load_model(args.checkpoint, config_section=shape,
                       attn_res_config={"attn_res": True,
                                        "attn_res_boundaries": list(presets.DEFAULT_BOUNDARIES[shape])},
                       device=args.device, meta=meta,
                       overrides={"rope_fp32": True, "attn_impl": "sdpa", "use_liger": False}).float().eval()
    num_layers = int(model.config.num_layers)
    if num_layers != preset.num_layers:
        raise SystemExit(f"{key}'s registry row has {preset.num_layers} layers; the model has {num_layers}")
    return model, AutoTokenizer.from_pretrained(presets.TOKENIZER_NAME), lane, meta


def _replay_states(model, ids: list[int], num_layers: int, device: str):
    """One sequence's states ``[layer, position, unit]`` (float32, on the device), in the
    hidden-state convention the fit reads: the eager capture forward, each layer's attention
    weights dropped as soon as they are computed and no per-layer keys or values kept, so a
    long stream fits in memory with the arithmetic unchanged."""
    import torch

    from kotodama.model.capture import cached_forward, hidden_rows

    x = torch.tensor([ids], device=device)
    with torch.no_grad():
        _, cap, _ = cached_forward(model, x, past=None, position_offset=0, light=False,
                                   capture_hidden=True, attention_sink=lambda _layer, _weights: None,
                                   capture_layers=set())
        stacked = torch.stack([h[0] for h in hidden_rows(cap, num_layers)])
    del cap
    return stacked


def _append(args: argparse.Namespace, *, preset, shape: str, prompts, settings, cf) -> None:
    """``--append_from``: sample (or one shard of) the long streams; then fit the tail and splice."""
    import hashlib
    import json
    import shutil

    import numpy as np
    import torch
    from anamnesis.extraction.calibration import (
        PCA_MODEL_NAME,
        POSITION_COUNT_FLOOR,
        POSITION_COUNTS_KEY,
        POSITIONAL_MEANS_KEY,
        POSITIONAL_MEANS_NAME,
        positions_calibrated,
    )
    from anamnesis.extraction.replay.manifest import ReplayManifest

    from kotodama.model.capture import generate_light

    base, out = Path(args.append_from), Path(args.out_dir)
    if base.resolve() == out.resolve():
        raise SystemExit("--append_from writes a NEW directory; --out_dir is the base itself")
    base_tokens_path = (base / PCA_MODEL_NAME).with_suffix(cf.TOKENS_SUFFIX)
    for need in (base / POSITIONAL_MEANS_NAME, base / PCA_MODEL_NAME, base_tokens_path):
        if not need.exists():
            raise SystemExit(f"the base calibration lacks {need}")
    if (out / POSITIONAL_MEANS_NAME).exists():
        raise SystemExit(f"{out / POSITIONAL_MEANS_NAME} already exists; an append never overwrites")
    if args.max_new_tokens is None:
        raise SystemExit("--append_from needs --max_new_tokens: the length of the tail's streams")
    with np.load(base / POSITIONAL_MEANS_NAME) as z:
        base_means = np.asarray(z[POSITIONAL_MEANS_KEY])
        base_counts = np.asarray(z[POSITION_COUNTS_KEY]) if POSITION_COUNTS_KEY in z.files else None
    base_reach = positions_calibrated(base_means, base_counts)
    stream_dir = out / "append_streams"
    stream_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, lane, meta = _load(args, args.model, shape, preset)
    num_layers = int(model.config.num_layers)
    top_p = settings.top_p if settings.top_p < 1.0 else None
    eos = None if args.suppress_eos else list(settings.eos_token_ids)

    def have() -> dict[int, dict]:
        rows: dict[int, dict] = {}
        for f in sorted(stream_dir.glob("streams-*.json")):
            body = json.loads(f.read_text())
            if body["lane"] != lane or body["max_new_tokens"] != args.max_new_tokens \
                    or body["suppress_eos"] != bool(args.suppress_eos):
                raise SystemExit(f"{f} was sampled under another lane, budget or EOS rule")
            rows.update({int(k): v for k, v in body["rows"].items()})
        return rows

    def sample(indices, name: str) -> None:
        rows: dict[str, dict] = {}
        t0 = time.perf_counter()
        with torch.no_grad():
            for index in indices:
                prompt = cf.encode_prompt(tokenizer, prompts[index], chat_template=False)
                torch.manual_seed(index)
                o = generate_light(model, torch.tensor([prompt], device=args.device), args.max_new_tokens,
                                   do_sample=settings.do_sample, temperature=settings.temperature,
                                   top_p=top_p, eos_token_id=eos, output_hidden_states=False)
                rows[str(index)] = {"ids": [int(t) for t in o.sequences[0].tolist()],
                                    "prompt_length": len(prompt)}
                logger.info(f"append stream {index}: {len(rows[str(index)]['ids']) - len(prompt)} generated, "
                            f"{time.perf_counter() - t0:.0f}s")
        path = stream_dir / f"streams-{name}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"lane": lane, "max_new_tokens": args.max_new_tokens,
                                   "suppress_eos": bool(args.suppress_eos), "rows": rows}))
        tmp.rename(path)

    if args.shard is not None:
        index, count = cf.parse_shard(args.shard)
        todo = [i for i in cf.shard_range(len(prompts), index, count) if i not in have()]
        sample(todo, f"{index + 1}-of-{count}")
        return
    missing = [i for i in range(len(prompts)) if i not in have()]
    if missing:
        sample(missing, f"rest-{missing[0]}-{len(missing)}")
    streams = have()

    # the arithmetic gate: BASE's own streams, replayed and reduced as the tail is
    base_streams = ReplayManifest.model_validate_json(base_tokens_path.read_text())
    depth, width0, hidden = base_means.shape
    dev = torch.device(args.device)
    gate = torch.zeros((depth, width0, hidden), dtype=torch.float64, device=dev)
    gate_counts = np.zeros(width0, np.int64)
    for gid in base_streams.gen_ids():
        st = _replay_states(model, list(base_streams.entry(gid).input_ids[:-1])[:width0], num_layers,
                            args.device)
        gate[:, :st.shape[1]] += st.double()
        gate_counts[:st.shape[1]] += 1
        del st
    rebuilt = calibration_append.means_from(gate.cpu().numpy(), gate_counts, POSITION_COUNT_FLOOR)
    del gate
    if not np.array_equal(rebuilt, base_means):
        raise SystemExit(
            "BASE's own streams do not rebuild BASE's means byte for byte under this replay "
            f"(max abs diff {float(np.max(np.abs(rebuilt - base_means))):.3g}): a tail fit here "
            "would sit on different arithmetic")
    logger.info(f"arithmetic gate: {len(base_streams.gen_ids())} base streams rebuild the base means byte for byte")

    width = max(len(r["ids"]) - 1 for r in streams.values())
    sums = torch.zeros((depth, width, hidden), dtype=torch.float64, device=dev)
    halves = torch.zeros((2, depth, width, hidden), dtype=torch.float64, device=dev)
    counts = np.zeros(width, np.int64)
    half_counts = np.zeros((2, width), np.int64)
    probe_start = max(0, base_reach - calibration_append.SEAM_SPAN)
    probe_stop = base_reach + calibration_append.SEAM_SPAN
    probes: dict[int, np.ndarray] = {}
    for n, i in enumerate(sorted(streams)):
        st = _replay_states(model, streams[i]["ids"][:-1][:width], num_layers, args.device)
        s_len = st.shape[1]
        st64 = st.double()
        sums[:, :s_len] += st64
        halves[n % 2, :, :s_len] += st64
        counts[:s_len] += 1
        half_counts[n % 2, :s_len] += 1
        if len(probes) < 6 and s_len > probe_stop:
            probes[i] = st[:, probe_start:probe_stop].cpu().numpy()
        del st, st64
    floor = int(args.count_floor)
    new_means = calibration_append.means_from(sums.cpu().numpy(), counts, floor)
    even = calibration_append.means_from(halves[0].cpu().numpy(), half_counts[0], floor)
    odd = calibration_append.means_from(halves[1].cpu().numpy(), half_counts[1], floor)
    del sums, halves
    means, pos_counts = calibration_append.splice(base_means, base_counts, base_reach, new_means,
                                                  counts, floor)
    seam = calibration_append.seam_report(base_means, base_reach, new_means, even, odd, means,
                                          probes, probe_start)

    np.savez_compressed(out / POSITIONAL_MEANS_NAME,
                        **{POSITIONAL_MEANS_KEY: means, POSITION_COUNTS_KEY: pos_counts})
    for name in (PCA_MODEL_NAME, base_tokens_path.name, f"{Path(PCA_MODEL_NAME).stem}.build.json"):
        if (base / name).exists():
            shutil.copyfile(base / name, out / name)

    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    if sha(out / PCA_MODEL_NAME) != sha(base / PCA_MODEL_NAME):
        raise SystemExit("the copied basis is not the base's bytes")
    record = None
    try:
        from anamnesis.extraction.calibration_extension import write_calibration_extension
    except ImportError:
        logger.info("the installed anamnesis records no calibration extensions; the appended "
                    "directory carries its own digest")
    else:
        record = write_calibration_extension(base, out).model_dump(mode="json")
    receipt = {
        "base_dir": str(base), "base_reach": int(base_reach),
        "positions_calibrated": int(positions_calibrated(means, pos_counts)),
        "tail_count_floor": floor, "n_streams": len(streams), "width": int(width),
        "streams": {"max_new_tokens": args.max_new_tokens, "suppress_eos": bool(args.suppress_eos),
                    "seeds": "torch.manual_seed(prompt index)", "temperature": settings.temperature,
                    "top_p": settings.top_p},
        "arithmetic": "base streams rebuild the base means byte for byte under this replay "
                      "(eager capture forward, float32 states, float64 sums)",
        "seam": seam, "lane": lane, "checkpoint_step": meta.get("step"),
        "calibration_extension": record,
    }
    (out / "calibration_append.json").write_text(json.dumps(receipt, indent=1))
    logger.info(f"appended: reach {base_reach} -> {receipt['positions_calibrated']}; seam new-vs-base "
                f"{seam['new_vs_base_rel']['median']:.4g} against split-half noise "
                f"{seam['split_half_noise_rel']['median']:.4g}")


if __name__ == "__main__":
    main()
