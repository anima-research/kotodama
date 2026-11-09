# Kotodama usage reference

The operating reference for the `kotodama` package: one command per task (golden
paths), the config system, the layout of training runs, the in-loop evals, the
binding batteries, the source layout and the tests. The project overview, the
architecture and the setup are in the [README](../README.md).

## Golden paths

One command per task. `tools/run_py.sh <script.py | -m module> ...` runs any
single-process job inside the launcher environment.

### Data

| task | command |
|---|---|
| NCA token stream (3B recipe) | `tools/nca_gen.sh` (defaults: 3500 rules × 20 sims, seed 17; split into train/eval and verified) |
| NCA v3 stream + eval sets | `python -m kotodama.nca.gen_dyadic_data --out <dir> ...` (token-level mix, banded rules, per-episode sidecar), then `--finalize --out <dir>` to concatenate and emit the eval sets |
| repeated-bigram statistics | `python scripts/utils/bigram_stats.py name=<stream.bin> [...] --out <json>` (in-context repeated-bigram opportunity and reliability over uint16 streams) |
| curate: ingest a source | `python -m kotodama.data.curation.ingest {--preset <name> \| --hf <id> --name <src> \| --local <path> --name <src> \| --spec <src.yaml>} --out_dir <cleaned>` (cleaning, quality filters, chunked JSONL, per-reason stats; `--list_presets`; `--adapter module:function` for a site-local source) |
| curate: deduplicate | `python -m kotodama.data.curation.dedup --in_dir <cleaned> --out_dir <deduped> [--mode minhash\|exact] [--priority <src>=N ...]` (across sources; the higher-priority copy survives) |
| curate: tokenize | `python -m kotodama.data.curation.tokenize --in_dir <deduped> --out_dir <tokenized>` (one uint16 `<src>.bin` per source, EOS after each document) |
| curate: assemble | `python -m kotodama.data.curation.assemble --in_dir <tokenized> --out <train.bin> --seq_len 4096 [--drop <src>] [--weight <src>=W] [--cap <src>=TOKENS]` (shuffled in `seq_len` chunks; `train.provenance.json` names every chunk's source) |
| curate: audit a source | `python -m kotodama.data.curation.audit --in_dir <chunks> --source <src> --class <web\|prose\|code\|...> --out_dir <dir>` (what the filters would drop or repair, with exemplars) |
| inspect / sample / remix a stream | `python -m kotodama.data.stream {inspect,sample,remix} <train.bin> ...` (size, documents and source shares; decoded chunks by source; a new stream with sources dropped or reweighted, and its own provenance) |

The curation stages run in order, and each stage reads the directory that the
previous stage wrote. A stage writes every file under a temporary name and
renames it when complete. A source or chunk that is already finished is skipped
on a rerun. The provenance file beside an assembled stream records the order of
its sources, their chunk counts and the shuffle seed. That record is enough to
label every chunk of the stream with its source (`kotodama.data.provenance`).
`.jsonl.zst` chunks need the `curation` extra.

### Train

| task | command |
|---|---|
| pretrain (NCA, language, CPT) | `tools/run_train.sh --config configs/<run>.yaml [--flag value ...]` (torchrun over `KOTODAMA_NPROC` GPUs, default 8) |
| 1B trunk | `tools/run_train.sh --config configs/tsumugi-1b.yaml --global_batch_tokens 2097152 --data_path <train.bin> --mix_data_path <mix.bin> --checkpoint_dir <dir>` |
| cooldown branch at anchor step S | copy `<dir>/anchors/step_<S>.pt` (8-digit step) into a fresh `--checkpoint_dir` and launch there with `--total_steps S+1+K --decay_start_pct (S+1)/(S+1+K) --save_schedule 1B:end --anchor_tokens ""` |
| in-loop evals | add `--inloop_evals riders,induction` to a train launch |

### Post-train

| task | command |
|---|---|
| chat SFT | `tools/run_py.sh -m kotodama.posttrain.sft --config configs/posttrain/tsumugi-sft.yaml --checkpoint <base.pt> --train_jsonl <chat.jsonl> --out_dir <dir>` (rows `{"messages": [...]}`; assistant turns train) |
| LoRA adapter | the chat SFT command plus `--lora_rank <r> [--lora_layers all\|prelude\|core\|coda\|<ranges>]`: trains an adapter on the frozen model and saves `checkpoints/step_*.adapter/` (`kotodama.posttrain.lora`; an adapter on the looped core applies on every pass) |
| DPO / DPOP / IPO | `tools/run_py.sh -m kotodama.posttrain.pref --config configs/posttrain/tsumugi-dpo.yaml --checkpoint <init.pt> --data <pairs.jsonl> --out_dir <dir>` |
| KTO against self-copying | `python -m kotodama.posttrain.selfcopy label --in <samples.jsonl> --out <kto.jsonl>`, then `pref` with `configs/posttrain/tsumugi-kto-repetition.yaml` |
| instruction data → transcripts | `python -m kotodama.posttrain.transcript convert-messages --in <*.parquet> --out <transcripts.jsonl>` (`User:` / `Assistant:`; `convert-markdown` for `**Name:**` dialogues) |
| self-play seeds | `python -m kotodama.posttrain.seeds --endpoint <url> --n 2000 --out <seeds.jsonl>` (the model writes each cast and setting) |
| self-play transcripts | `python -m kotodama.posttrain.selfplay --seeds <seeds.jsonl> --endpoint <url> --out <run>` (K candidates per turn, mechanical vetoes, uniform choice among survivors, every candidate banked) |
| self-play → SFT data / health / coherence study | `python -m kotodama.posttrain.corpus {assemble,meters,precision-sample,rate,score} <run> ...` |
| sampling-law bench (sampler dials and steering dials on fixed panels) | `python -m kotodama.posttrain.lawbench {stage-a,generate,fluency,damage-set,summary} ...` against a gateway fleet (cells share batches: per-row laws and levers; seeds shared across cells), then the damage readout `tools/run_py.sh scripts/eval/law_damage.py --cells <run>/cells.jsonl --set <run>/damage_set.jsonl ...` (each cell's NLL of a fixed continuation set under its whole law, teacher-forced) |

Conversations are plain-text transcripts with no special tokens: an optional prose header, then `Name: text`
turns separated by blank lines (`kotodama.posttrain.transcript`); `--train_speakers all` puts loss on every turn.
`kotodama.posttrain` trains single-GPU on pretraining's precision recipe (bf16
weights, fp32 masters in Muon + AdamW) with fp32 AttnRes routing, the numerics the
serving engine computes. Preference runs precompute reference log-probs once (disk-cached, verified on
reuse) and refuse more than one epoch. The self-copy labeller flags verbatim reuse
of the model's own text and template loops, never reuse of names or objects.

### Evaluate

`<ckpt>` is a `.pt` / `.pt.zst` path, a registry name (`configs/checkpoints.yaml`), a published-weights directory, or
`hf:REPO[@REVISION]` (e.g. `hf:LuxiaSL/kotodama-tsumugi-1b`, or a step branch of its trajectory repo). `<preset>` is a
`kotodama.presets` shape (`3b`, `1b-tsumugi`, `proxy`, …). Every Kotodama-checkpoint eval loads through
`kotodama.eval.model_loader`, which takes the AttnRes boundaries from the published `config.json`, else the preset's
trained boundaries; pass `--attn-res-boundaries` only to override them. Public Hugging Face models have their own
runner per eval, scored with the same math.

| task | Kotodama checkpoint | public HF model |
|---|---|---|
| lm-eval battery | `EVAL_SECTION=<preset> tools/launch_lmeval_sharded.sh <ckpt> 0,1,2` (task-sharded over 3 GPUs) · one GPU: `tools/run_py.sh -m scripts.eval.run_lm_eval --checkpoint <ckpt> --config-section <preset>` | `tools/run_py.sh -m scripts.eval.run_lm_eval_hf --model <repo> [--revision <rev>] --output-dir <dir>` |
| Uncheatable BPB | `tools/run_py.sh -m kotodama.eval.uncheatable --checkpoint <ckpt> --config-section <preset> --out <json>` (data: `--data-dir`, else `$KOTODAMA_UNCHEATABLE_DATA`, else the HF dataset; checkpoint sweeps: `--max-docs-per-domain N`) | `tools/run_py.sh -m scripts.eval.run_uncheatable_hf --model <repo> [--revision <rev>] --out <json>` |
| Uncheatable BPB through vLLM (hybrid and state-space models, which vLLM runs with fused kernels) | — | `KOTODAMA_VENV=<vllm venv> tools/run_py.sh -m scripts.eval.run_uncheatable_vllm --model <repo> [--tp N] [--trust-remote-code] --out <json>` (same data, chunking and reduction as the HF runner; per-token losses are vLLM's fp32 log-probs) |
| induction score | `tools/run_py.sh -m kotodama.eval.induction --checkpoint <ckpt> --config-section <preset> --tag <t>` | — |
| NCA gates | `python -m kotodama.eval.nca_gates --checkpoint <ckpt> --eval-dir <dir> --tag <t>` · every checkpoint of a run: `tools/run_py.sh -m scripts.eval.nca_gates_sweep --ckpt-dir <run> --eval-dir <dir> --init-seed <seed>` | — |
| binding suite: score | `tools/run_py.sh -m kotodama.eval.binding run --model koto:<ckpt> --name <m> --batteries <battery or group> --out <dir>` (one model load for every battery; `list` shows batteries and the groups `scorecard`, `bindprobes`, `fluid`, `formupdate`, `all`) | the same with `--model hf:<repo or dir>` |
| binding suite: read | scorecard `python -m kotodama.eval.binding report --scores <dir>` · capacity size law `… capacity --scores <dir> --params <tsv>` · fluid vs crystallized `… fluid --scores <dir>` · form vs update `… formupdate --scores <dir>` · bindprobes batteries `python -m binding_bench.bindprobes aggregate --grids <dir>/<battery>` | same |
| binding suite: regenerate the pinned prompts | `python -m kotodama.eval.binding gen-capacity --out <file>` · `gen-fluid --out <dir>` · `gen-formupdate --out <dir>` (byte-identical to the pins) | — |
| rider grid (not in the suite) | `python -m binding_bench.riders {gen,run,aggregate}` (scored against a server's `/logprobs`) | same, against any server |
| new bindprobes item sets | `python -m binding_bench.bindprobes gen --out <dir> [--families setbind,answer,adjacency,codevarmut] --verify`, then `run` (against a server) / `aggregate` · deixis validity gates: `… gates --grids <dir> --items items.jsonl --chance-model <m> --learn-model <m> [--pilot]`. The frozen sets are the suite's `bp-*` batteries | same |
| AttnRes boundaries from one checkpoint | `tools/run_py.sh scripts/eval/nca_profile.py --checkpoint <ckpt> --config-section 1b --out <json>` | — |
| AttnRes boundaries over a run | `tools/run_py.sh scripts/eval/fit_boundaries.py --run <label> --checkpoint <run>/weights/step_*.pt --config-section 1b --out-dir <dir>`, then `--summarize --run <label> --out-dir <dir>` | — |
| AttnRes routing matrix | `tools/run_py.sh scripts/eval/routing_matrix.py --run <label> --checkpoint <ckpt> [--checkpoint ...] --config-section 1b --attn-res-boundaries dd3b --out-dir <dir>` (every routing site × source on held-out text, per domain; `--plot` renders finished JSONs) | — |
| capture / readout | `kotodama.model.capture` (instrumented AttnRes forward, parity-tested against `llama.py`) | — |

### Serve

| task | command |
|---|---|
| one model | `tools/run_serve.sh --checkpoint <ckpt> [--mode chat] [--model_size <shape>]` |
| one model with a LoRA adapter | `tools/run_serve.sh --checkpoint <base> --adapter <dir>.adapter ...` (the adapter is merged into the weights at load. The served model id gains `+<dir name>` without `.adapter`, and `/info` reports the adapter's rank, alpha, module count and digest. Steering calibrated on the base model, `--steer-npz` and `injection` requests, is refused) |
| the public 3B base | `tools/serve_3b_base.sh` (downloads the checkpoint once, then serves it) |
| the released kotodama-tsumugi 1B base | `tools/run_serve.sh --profile tsumugi-1b-quick` (`configs/serve/tsumugi-1b-quick.yaml`: the published weights from the Hub on any one GPU, compiled without kernel benchmarking so it starts in minutes, two batched row buckets, cascade attention, the prefix KV pool, per-request steering, seeded candidates, top-p / min-p; any flag after it overrides the profile, e.g. `--checkpoint <dir>`, `--port`, `--steer-npz <banks>`; requests set sampling: the `tsumugi` law in `kotodama.serve.laws` is temperature 0.75, top-p off, repetition 1.2) |
| tsumugi under sustained load | `tools/run_serve.sh --profile tsumugi-1b` (four row buckets up to 128 rows, max-autotune kernels: a long first compile, then more throughput; `--compile_store <dir>` or `$KOTODAMA_COMPILE_STORE` keeps the compiled artifacts so later starts with the same code, runtime and GPU load them) |
| a fleet behind one port | `tools/run_gateway.sh --config configs/gateway-tsumugi.example.yaml` (`kotodama.serve.gateway`: spawns and supervises one replica per GPU from a serve profile, routes each conversation to the replica holding its prefix cache, else the least loaded, and serves completions, chat, `/generate`, `/score_tokens`, `/logprobs` and `/info` on one port; clients pin a conversation explicitly with the OpenAI `user` field; `configs/gateway.example.yaml` is a single 3B replica) |
| chat / completion client | `python -m kotodama.serve.chat [complete]` (sampling laws: `kotodama.serve.laws`) |
| per-request steering | on in the tsumugi profile (`--levers <slots>` elsewhere, with `--steer-npz <banks>` for named vectors): every steered request carries its own lever (bank `vectors`, or a raw `injection` at an attention or MLP site), `steer_span` `decode` or `uniform` (the prompt too), and steered and unsteered requests batch together (`kotodama/model/levers.py`) |
| signatures of generated rows | start the server with `--sign_spool <dir>`, plus `--kernel_pins <table>` so that every process runs the same kernel configs. A request that sets `sign` (on `/generate`, `/v1/completions` or `/v1/chat/completions`) leaves one record per candidate in the spool, and each choice carries its `sign_id`. `tools/run_py.sh -m kotodama.serve.sign --spool <dir> --checkpoint <ckpt> --reducer <package.module:attr> --layers <L,...> --pins <table>` signs the records: it replays each one through a signing decoder and passes the signed rows to the reducer (`kotodama/model/signer.py`). A server started with `--sign_reducer` also signs its own spool whenever it is idle. With `--sign_inflight` (and `--xbatch_rows` with `--cascade_attn`), the server signs a request while it generates it: the candidates decode on a signing decoder with the widest bucket's geometry, so they get the tokens an unsigned request gets. The server reduces each candidate's span itself, and the signatures land in the spool as signed batches. `tools/run_py.sh -m kotodama.serve.sign_archive archive --spool <dir> --to <root>` copies a spool's signed batches into compressed segments on long-term storage, one zstd frame per batch, with the audit captures in a stream of their own. `extract` writes a batch back to its plain directory. The spool's own copy is removed only by `drop_local`, after the archived bytes are read back and match it; `/replay` then reads the batch from the archive (`kotodama/serve/sign_archive.py`) |
| a log of every request | start the server with `--request_log <dir>` (off by default; on a gateway fleet, `serve_args: ["--request_log", "<dir>"]` turns it on for every replica). Each generation or scoring request becomes one JSON line: the request as sent (the caller's `user` included), the response, latency, the weights digest, and each candidate's exact prompt and generated token ids, stop token and `sign_id`. Files are daily (`<dir>/<YYYY-MM-DD>/<writer>.jsonl`, UTC) and are compressed once their day is over. A request that sends `"log": false` is not written, and client addresses are never recorded (`kotodama/serve/request_log.py`) |
| a prompt's KV, however it was chunked or cached | on by itself on CUDA with flash-attn installed (`DecodeEngine.block_invariant`). Prefill attention runs flash-attn's causal KV-cache kernel with one split, so every row reduces over its keys in one fixed order. Every prompt and extend runs through compiled blocks of one width (256 tokens by default, `--prefill_buckets`), with the last chunk padded on the right, so no cached position is forwarded again. A prompt's KV and logits then come out bit for bit the same whether it was prefilled cold, extended from a prefix cache or pool hit, or extended in a KV session. Seeded and signed requests therefore read the caches too, from prompt KV only (`kotodama/model/regional.py`, `kotodama/model/decode_engine.py`) |
| a conversation whose KV lives across its turns | start the server with `--kv_sessions <n>` (and `--xbatch_rows`). `POST /kv_session/open` returns a session id. Each turn then sends only its new tokens with `kv_session` on `/v1/completions` or `/generate`, and names the candidate it continues with in `append_from`. The server prefills the new tokens against the KV the session holds and keeps the KV each candidate's decode steps wrote. Between turns, `/kv_session/evict` drops a range of positions and re-rotates the survivors through the model's RoPE table, and `/kv_session/export` and `/kv_session/import` take rows out and put them back at a position. A session's KV never reaches another request. Signed turns are signed in flight, each record carries `meta.cache_state`, and their lane names the session conventions. A gateway sends every request of a session to the replica that holds it (`kotodama/serve/kv_sessions.py`, `kotodama/model/session_kv.py`) |
| neuron steering sweep (one process per GPU) | `tools/run_py.sh scripts/sweep/steer_sweep.py --checkpoint <ckpt> --model_size 1b-tsumugi --rope_fp32 --dose_norms <p50.json> --mode decode\|uniform --layers <L> --shard <i> --n_shards <n> --out <dir>`, then `scripts/sweep/merge.py --sweep <dir> --mode <mode>` (per-layer run files a neuron explorer ranks) |

The server exposes `/v1/chat/completions`, `/v1/completions`, `/generate` (full
sampling control, streaming), `/logprobs` (batched prefill scoring, the
binding batteries' scorer), `/v1/models`, `/info` and `/health`. The default
decode path is a static-KV-cache, compiled, CUDA-graphed engine
(`kotodama/model/decode_engine.py`); the eager `llama.py` loop is the parity
reference. ChatML and stop tokens live in `kotodama/serve/chatml.py`.

### Telemetry and tooling

| task | command |
|---|---|
| post per-checkpoint results to the metrics API | `python scripts/eval/hm_backfill.py --job <id> --gates <dir>` (Uncheatable JSONs: `--uncheatable <json>`) |
| read a metric family back | `python scripts/eval/hm_portrait.py --job <id> --x 'norm_gain/layer_{i}/q_norm' [--y 'geo/layer_{i}/attn_entropy_mean'] --out <png>` (heatmap without `--y`, trajectory with it; repeat `--job` to compare runs) |
| geometry diff of two runs | `python scripts/eval/hm_geodiff.py --job <A> --job <B> --out-dir <dir> [--max-step N]` |
| spike resilience | `python scripts/eval/resilience.py --job <id> [--signal train/grad_norm]` (per loss spike: whether and when `geo/rankme_last` recovers) |
| checkpoints | `python -m kotodama.ckpt {info,strip,list}` (names from `configs/checkpoints.yaml`) |
| node parity | `python scripts/utils/node_parity.py --hosts <h1> <h2> --mirror <path> --python <venv python>` (same stack on every machine before moving a run) |
| mirror to a machine without git | `tools/sync.sh [-n] [host:/path]` (exactly the tracked files; default target `$KOTODAMA_SYNC_TARGET`; the remote's `site.env`, venvs, caches, `logs/` and `analysis/` are kept) |

The telemetry readers talk to the scheduler metrics API at
`$KOTODAMA_SCHEDULER_URL`; `train.py` posts to it when a job id is present.

## Configs

Configs are flat YAML whose keys are `train.py` flags, with `extends:`
inheritance; CLI flags override config values. Shared blocks live in
`configs/base/` (`optim-muon`, `runtime-production`, `3b-lineage`,
`7b-throughput`). `tests/data/config_snapshot.json` pins every config's
effective values; it is regenerated on purpose with
`KOTODAMA_WRITE_CONFIG_SNAPSHOT=1`.

- **The 3B lineage** (`nca-3b-phase1` → `nca-3b-phase3-cotrain` → `3b-language`,
  plus `3b-chinchilla-decay`) is the record of the shipped run: NCA
  pre-pretraining without AttnRes, an NCA co-train with AttnRes on the
  boundaries derived from that run's geometry, then language pretraining from
  the NCA checkpoint with re-initialized embeddings (`--resume_nca`).
- **`ladder-*`** are 1B-budget arms: language arms extend `ladder-a0.yaml`
  (`1b` shape, DD-3B boundaries, 30B tokens; the `-deep*` arms use the 52-layer
  shapes), NCA-phase arms extend `ladder-a1-nca.yaml`, and `ladder-cf-*` are
  no-AttnRes counterfactuals for boundary fitting.
- **`tsumugi-1b.yaml`** runs the `1b-tsumugi` shape as a
  constant-LR trunk with anchors (see [Trunk and cooldown branches](#trunk-and-cooldown-branches)).
- **`canary-7b*`, `bench-7b-profile`, `soak-7b`** are 7B throughput configs
  (canaries: `tools/run_canary.sh 7b`); their uniform AttnRes blocks are a
  placeholder, not a geometry fit.

## Training runs, in-loop evals and the binding batteries

### Trunk and cooldown branches

A trunk trains at constant LR to a token ceiling and never decays
(`decay_start_pct: 1.0`). Two checkpoint streams come off it:

| flag | effect |
|---|---|
| `--save_schedule 250M:4B,1B:end` | weights-only saves on a piecewise token cadence → `<checkpoint_dir>/weights/` (never pruned) |
| `--anchor_tokens 10B,20B,...` | protected full (resumable) checkpoints → `<checkpoint_dir>/anchors/` |

An annealed read is a **branch** from an anchor. Copy the anchor into a fresh
directory and resume training there, from the newest `step_*.pt` in
`--checkpoint_dir`, with a short learning-rate decay that starts exactly at the
anchor step. The cooldown command under [Train](#train) does this. The trunk is never disturbed, so
one run yields annealed models at every anchor.

Other run-shape flags:

| flag | effect |
|---|---|
| `--stop_at_step N` | exit with a final checkpoint after step N−1; the LR and batch schedule are unchanged (`--total_steps` redefines them) |
| `--init_scheme` | `standard` (default) / `mimetic` / `orthogonal` / `depth_graded` / `small_attn_out`, applied over the standard init (`kotodama.model.init_schemes`) |
| `--mix_data_path`, `--mix_fraction`, `--mix_window START:END` | interleave a second token stream at an exact token fraction (token-debt scheduler, resumable), optionally only inside a global-token window |
| `--data_seed` | data-order seed, independent of the init `--seed` (default: `--seed`) |

### In-loop evals

`train.py --inloop_evals <suites>` scores small suites on the live weights
in-process during training. Every DDP rank scores a shard, rank 0 gathers; off
by default, and not available with `--tp_size > 1`.

| suite | what | needs |
|---|---|---|
| `riders` | riders-lite: `--inloop_riders_n` balanced items (default 200) × 4 registers × {0shot, 3shot, 3shot_relet}, scored by the same function as `/logprobs`; accuracy, chance-corrected accuracy, margin, error types | nothing (or `--inloop_riders_items` + `--inloop_riders_shots`) |
| `setbind` | set binding: every static item (no transfers) of a bindprobes items file, 3-shot, scored by the same function as `/logprobs` and read by the benchmark's `set_metrics`; `setbind/k1` is hit@1 on the one-object sets, plus `k2`, `k3`, `k23`, `k23_exactk`, margins and counts | `--inloop_setbind_items` |
| `induction` | random-token repeat loss drop | nothing |
| `uncheatable` | bits/byte on the first `--inloop_uncheatable_docs` documents per domain | `--inloop_uncheatable_path` |
| `nca_gates` | NCA ledger accuracy, for NCA-phase runs | `--inloop_nca_gates_dir` |

Cadence is `--inloop_schedule` (default `0:0.025,decay:0.005`: every 2.5% of
tokens, every 0.5% once the LR decay starts, plus the first and final steps). A
point fires on the same step rule as `--save_schedule`, so eval `step_N` and
checkpoint `step_N` hold the same weights. Each point posts one metrics batch
(`inloop/...` keys) and writes every row to
`<checkpoint_dir>/inloop/step_<N>.jsonl`; resumes skip points already written.
Size knobs: `--inloop_max_batch_tokens`, `--no_inloop_riders_relet`.

The `setbind` items are the benchmark's `bp-v3.1` battery (64 one-object sets
and 256 sets of two or three objects among its static items). The file is
pinned by sha256 in the benchmark's battery registry; when the file matches a
pin, the suite also checks that its prompts render to the battery's render pin,
and refuses to start otherwise. `bp-v3.1` comes from an earlier version of the
setbind generator, so it is rebuilt from its recipe (seed 2031) in a
[binding-bench](https://github.com/anima-research/binding-bench) checkout,
which writes `DIR/bp-v3.1/items.jsonl` and checks it against the pin:

```bash
python -m tools.replay_recipe --kotodama /path/to/kotodama-clone --battery bp-v3.1 --out DIR
train.py ... --inloop_evals riders,setbind --inloop_setbind_items DIR/bp-v3.1/items.jsonl
```

### Binding batteries

The binding batteries belong to the binding benchmark (the `binding-bench`
package, installed by the `bench` extra); `python -m kotodama.eval.binding`
runs its suite with kotodama checkpoints loadable (`kotodama.bench_plugin`).
The batteries are judge-free: each item has a closed candidate set of
single-token answers, and the read is the logprob of each candidate at the
answer position, scored through the server's `/logprobs`
(`kotodama.serve.logprobs`, which matches the benchmark's own scorer). Chance is 1/C for C candidates;
aggregates report chance-corrected accuracy, (acc − chance) / (1 − chance),
alongside margin (gold logprob minus the best distractor) and typed errors.
Every item also runs under few-shot conditions with exemplars.

- **Rider grid** (`binding_bench.riders`): entity-state tracking. N boxes go
  through M operations (swap, move, put, overwrite) and one box is queried;
  the same abstract item is rendered in four registers (Python code, narrative
  prose, screenplay, chat). Candidates are the objects mentioned, so chance is
  1/C per item; the gold is never the most recently mentioned object outside a
  recency-control subset. A difficulty-0 copy tier (1 box, 0 ops, 8 candidates)
  gates harness validity per register. The `3shot_relet` condition re-letters
  the exemplars' boxes to separate following a procedure from copying labels.
- **Bindprobes, core** (`binding_bench.bindprobes`): the same question in
  other shapes, each under `0shot`, `fair`, `3shot` (collision-free exemplars)
  and `3shot_collide` (exemplars re-labeled into the item's own names, different
  bindings). An item's own three exemplars come from its own cell, so in some
  families they show where the answer sits. `fair` is the standard few-shot
  reading: four exemplars from the whole battery, balanced on every cell
  component, so they show the task's format but not the answer's position.
  - `attr`: N ∈ {2, 4, 6, 8} people each hold one object, plus unbound
    distractors; one is queried.
  - `codevar`: variable binding in Python with copies, reassignments and swaps;
    candidates are every literal in the program.
  - `dialogue`: multi-party prose in which objects change hands, partly through
    pronouns; query the holder of an object or the object of a holder.
  - `deixis`: a chat transcript where ownership is stated only through
    I / me / you / your; at least 3 candidates (chance ≤ 1/3); each item comes
    with a third-person twin and, at three speakers, a speaker-transposed twin,
    so the deixis penalty and twin consistency are read off the pairs.
    `gates` runs the validity gates V1–V5.
- **Bindprobes, v3 families** (generated only when named):
  - `setbind`: one person holds a set of k ∈ {1, 2, 3} objects among 8 in
    play; hit@1 (chance k/8), exact@k (chance 1/C(8, k)) and set margin.
  - `answer`: production-side deixis: a speaker names the holder as they would
    say it (`me` / `you` / a name; chance 1/4), with a body-hidden null variant
    for contextual calibration.
  - `adjacency`: a vocative-free "you" at three speakers, resolved by request →
    grant adjacency (3 names, chance 1/3).
  - `codevarmut`: Python list aliasing with mutate / rebind / copy traps; 5
    literals per item (chance 1/5); only alias-graph tracking clears all three
    traps.

- **The binding suite** (`kotodama.eval.binding`): one runner for every battery
  above and below. It loads a model once (a kotodama checkpoint or any Hugging Face
  model), scores each requested battery in-process with the same scorer as
  `/logprobs`, and writes resumable rows. Prompt files are pinned by their sha256. A
  bindprobes item file is also pinned by the sha256 of every prompt that it renders,
  because each item renders with the templates of the version that generated it.
  So every recorded number traces to the exact strings that it was scored on. Beyond the bindprobes families
  it carries:
  - stage gauges `joinlite-v1` (static I / you and a speaker-swap counterfactual)
    and `swapdyn-v1` (the swap on transfer dialogues). A shift above +.02 at the
    low end of its 95% CI binds; one below −.02 at the high end inverts.
  - `stage5-decomp-v1`: dynamic "you" split into its prerequisites (third-person
    possession transfer, deictic transfer speech acts, query-time deixis), each
    a prompt variant over the same episodes.
  - `capacity-v1`: binding capacity after arXiv 2609.30634. K "Name: task" lines
    (K = 1…64), with or without ~256 tokens of filler, then one query. K50 is
    where recall falls halfway to chance; `capacity` fits K50 ∝ N^α over
    baselines scored on this same harness and places any model on the law as a
    parameter-equivalent size.
  - the `fluid` group: four batteries whose answers exist only in the prompt
    (`fluid-v1-morph` invented inflection, `fluid-v2-algebra` invented
    operators, `fluid-v2-rebind` reassignment chains in prose and code with a
    stale-value rate, `fluid-v1-seqind` abstract token patterns) and a
    crystallized control in the same few-shot frame (`cryst-v1`: capitals,
    chemical symbols, books, currencies). Each family climbs a difficulty
    ladder and is read by the ladder position where accuracy falls halfway to
    chance; `fluid` z-scores fluid and crystallized ability across the models
    it is given. The v2 algebra and rebind files control two shortcuts the v1
    files leave open (copying a query operand; recency over the candidates);
    `fluid-v1` names the v1 set, and `fluid --set v1` reads it.
  - `formupdate-v2`: forming a binding vs updating it. Each base item binds K
    holders to values and asks about one; three conditions differ only in the
    U lines before the query (neutral filler, updates to other holders, or one
    update to the queried holder). `formupdate` reads form accuracy, the
    interference cost and the update cost, raw and normalized by headroom, with
    paired-bootstrap intervals over base items, and the stale-value rate. v2
    matches where the gold sits among the candidates mentioned (first, middle or
    last) across the three conditions, and reports the costs on the items where
    it is a middle mention in all three, so a primacy or recency bias moves no
    cost there; `formupdate-v1` keeps the earlier layout (`formupdate --battery`).
  `report` prints one scorecard row per model.

Rider generation is byte-identical per seed and name pool, so banked grids stay
comparable. Shortcut controls (recency, last speaker, last literal) are
scheduled at the chance rate and audited.

## Layout

```text
src/kotodama/
  model/       llama.py (model, AttnRes, looped depth), flash_attn_res/ (Triton kernels),
               decode_engine, capture, init_schemes, routing_read
  nca/         NCA trajectory generator, dyadic variant, v3 dataset writer
  training/    train.py, Muon, checkpointing, inloop (in-loop evals)
  data/        uint16 token-stream dataset, provenance, stream (inspect / sample / remix),
               curation/ (ingest, clean, filters, dedup, tokenize, assemble, audit)
  eval/        model loading, lm-eval adapter, uncheatable, induction, nca_gates,
               riders/ (rider grid), bindprobes/ (binding probes), binding/ (the suite: one runner + scorecard)
  monitoring/  per-layer geometric telemetry, wandb callback
  serve/       server, gateway, chatml, laws, steering, chat client
  presets.py   shapes, AttnRes boundaries, tokenizer
  ckpt.py      checkpoint load / strip / info / registry
configs/       training runs (+ base/), gateway example, checkpoint registry
scripts/       eval/ (lm-eval, boundary fitting, routing, telemetry readers),
               benchmark/ (serving performance and parity batteries),
               utils/ (kernel parity, data, smoke test)
tools/         launchers (all via _env.sh)
tests/         CPU test suite
```

## Tests

```bash
.venv/bin/python -m pytest
```

The suite runs on CPU; GPU-only tests skip without CUDA. Kernel parity checks
(`scripts/utils/*_parity.py`) and serving batteries
(`scripts/benchmark/check_*.py`) run standalone on a GPU.
