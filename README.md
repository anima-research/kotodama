<p align="center"><b>言霊 紬</b><br><sub>kotodama · tsumugi</sub></p>

> **言霊 kotodama** — word-spirit: language as a thing with a life of its own.
>
> **紬 tsumugi** — spun thread: words woven into more than the sum of their parts.

# Kotodama

> Led by [Luxia](https://x.com/kataletheia), with [Anima Labs](https://animalabs.ai/) · built with
> [Claude](https://www.anthropic.com/claude) · project pages:
> [kotodama.animalabs.ai](https://kotodama.animalabs.ai/)

Kotodama is a stack for pretraining language models from scratch, in one Python
package, `kotodama`. It trains Llama-style decoders with **Block Attention
Residuals** (AttnRes), placing the block boundaries from the network's own layer
geometry. Matrix parameters are optimized with **Muon**. A run of layers can
execute several times with shared weights (**looped depth**), and training can
begin on **neural-cellular-automata (NCA) trajectories** before language.
Around the trainer sit an evaluation suite (lm-eval, Uncheatable BPB, induction,
judge-free binding batteries), per-layer telemetry, and an OpenAI-compatible
serving stack.

The released model is
[kotodama-tsumugi-1b](https://huggingface.co/LuxiaSL/kotodama-tsumugi-1b): a
1.42B base model run as 72 effective layers (a 20-layer core executed twice) with
AttnRes boundaries fitted to its layer geometry, trained on 110B tokens. It was
built around one question: whether a model reading *you* in a conversation knows
who *you* is. Its full checkpoint trajectory, including every annealed branch, is
published at
[kotodama-tsumugi-1b-trajectory](https://huggingface.co/LuxiaSL/kotodama-tsumugi-1b-trajectory)
(and a second seed at
[kotodama-tsumugi-1b-seed7-trajectory](https://huggingface.co/LuxiaSL/kotodama-tsumugi-1b-seed7-trajectory)),
and the [Binding Atlas](https://kotodama.animalabs.ai/binding-atlas) places it
among public models on referent-binding tests.

## What tsumugi shows

The [Binding Atlas](https://kotodama.animalabs.ai/binding-atlas.html) tests whether a model tracks who *you* is, with judge-free
batteries scored against public base models. On those tests, kotodama-tsumugi-1b:

- **binds a static *you* early.** It binds *you* to the person being addressed by 33B
  training tokens. OLMo-2 13B needed about the same number of tokens for this stage;
  OLMo-2 1B needed about 420B ([stage cost by model size](https://kotodama.animalabs.ai/binding-atlas.html#f3h)).
- **follows hand-offs told in the third person.** When objects change hands in a
  story, it tracks who holds them better than every Pythia model up to 6.9B and than
  Llama-3.2-1B ([stage 5, taken apart](https://kotodama.animalabs.ai/binding-atlas.html#t1h)).
- **keeps *you* apart from the speaker.** Read zero-shot, its interpretation of a *you*
  whose referent has moved isn't inverted toward the speaker.
- **tracks *you* through hand-offs from three examples.** Shown three worked examples,
  it tracks who *you* is after objects change hands. Among public base models at 3B
  or below, it is the only one that does. The next smallest model that does is
  Qwen3-4B, which was trained on 36T tokens, about 300 times tsumugi's 110B.

In short, it binds like base models of 1–2B parameters trained on 40 to 300 times
more data.

## Architecture

- **Decoder:** grouped-query attention, RoPE (θ = 500K), RMSNorm with QK-norm,
  SwiGLU feed-forward, tied input/output embeddings, z-loss on the logits.
- **Tokenizer:** SmolLM2 (`HuggingFaceTB/SmolLM2-135M`, vocabulary 49,152).
- **Block Attention Residuals** ([Moonshot AI, 2026](https://arxiv.org/abs/2603.15031)): they replace the single
  running residual sum with blocks of layers. Each sublayer reads a mixture of
  the finished blocks' outputs and the current block's partial sum. A learned
  query at each layer sets the softmax weights of that mixture. Fused Triton
  kernels for the two routing phases are vendored in
  `kotodama/model/flash_attn_res/`; a PyTorch fallback runs on CPU.
- **Muon** (momentum orthogonalized by Newton–Schulz) for hidden matrices,
  AdamW for embeddings, norms and scalars (`kotodama/training/muon.py`,
  recipe in `configs/base/optim-muon.yaml`).
- **Looped depth** (optional): see [Looped depth](#looped-depth).

Model shapes, their default AttnRes boundaries and the tokenizer live in one
table, `src/kotodama/presets.py`. Every other model constant (RoPE θ, norm eps,
QK-norm, tied embeddings, z-loss weight) is a `KotodamaModelConfig` default in
`kotodama/model/llama.py`.

| shape | layers (effective) | width | notes |
|---|---|---|---|
| `3b` | 28 | 3072 | the earlier 3B release; DD-3B boundaries |
| `7b` | 32 | 4096 | throughput configs only |
| `1b` | 28 | 2048 | ~1.42B; DD-3B boundaries verbatim |
| `1b-deep` | 52 | 1536 | the `1b` parameter budget spent on depth |
| `1b-deep-loop` | 52 from 28 unique | 1536 | 12-layer core run 3× |
| `1b-deep-loopfit` | 52 from 42 unique | 1536 | 10-layer core run 2×; boundaries `CF52_FIT` |
| `1b-tsumugi` | 72 from 52 unique | 1536 | the released model; 20-layer core run 2×; boundaries `TSUMUGI_LOOP` |
| `intermediate`, `proxy`, `smoke` | 28 / 28 / 4 | 1024 / 512 / 256 | small-scale and test shapes |

## Setup

```bash
uv venv .venv
# torch first, built for your CUDA (the default wheel targets the newest CUDA); e.g. for CUDA 12.8:
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -e ".[eval]"                          # load, generate, evaluate
uv pip install --python .venv/bin/python -r requirements-serve.txt             # the OpenAI-compatible server
uv pip install --python .venv/bin/python -e ".[training,eval]" --no-build-isolation   # training: builds flash-attn against your torch (about 15 minutes)
cp site.env.example site.env   # optional: data root, venv, caches for this machine
```

The `nca` extra adds the JAX stack; `all` installs every extra. Python ≥ 3.12.

`site.env` (gitignored) holds everything machine-specific: `KOTODAMA_VENV`,
`KOTODAMA_DATA_ROOT` (token streams, NCA data, checkpoints),
`KOTODAMA_CACHE_ROOT` (persistent torch.compile / Triton caches),
`KOTODAMA_SCHEDULER_URL` and `KOTODAMA_SCHEDULER_JOB_ID_VAR` (metrics API),
`KOTODAMA_UNCHEATABLE_DATA`, `HF_HOME`. Every launcher in `tools/` sources
`tools/_env.sh`, which loads `site.env`, activates the venv, and exports the
package path and compile caches, so one machine's paths live in one file.

## Documentation

The full operating reference is [docs/USAGE.md](docs/USAGE.md):
- [golden paths](docs/USAGE.md#golden-paths), one command per task: data curation, training, post-training,
  evaluation, serving and telemetry;
- [configs](docs/USAGE.md#configs);
- [trunk and cooldown branches](docs/USAGE.md#trunk-and-cooldown-branches) and
  [in-loop evals](docs/USAGE.md#in-loop-evals);
- [the binding batteries](docs/USAGE.md#binding-batteries);
- [the source layout](docs/USAGE.md#layout) and [the tests](docs/USAGE.md#tests).

## How the distinctive pieces work

### AttnRes boundaries from layer geometry

Kotodama places block starts from the model's own geometry rather than
uniformly. The DD method works in three steps. It profiles a checkpoint
trained **without** AttnRes. It scores the transitions between adjacent layers by
attention entropy (and, in `fit_boundaries.py`, by linear CKA between their
outputs). Then it suggests where blocks should start, under explicit rules: the
embedding layer gets its own block, the last block keeps a few layers of cushion,
and the middle block stays contiguous. `scripts/eval/nca_profile.py` is the
archived algorithm that produced DD-3B, kept verbatim and pinned by a test.
`scripts/eval/fit_boundaries.py` fits every snapshot of a no-AttnRes run on
held-out language text and declares a fit settled at the first snapshot after
which its cuts never change. The resulting tuples (`DD3B`, `DDV1`,
`DD3B_DEEP52`, `CF52_FIT`, `TSUMUGI_LOOP`) are named in `presets.py`.

### Looped depth

Model config `loop_start` / `loop_len` / `loop_times`: a core of `loop_len`
layers starting at `loop_start` executes `loop_times` times with shared
attention and FFN weights. `num_layers` counts effective positions: AttnRes
boundaries, KV caches and capture see one layer per position, and each position
keeps its own pre-norms and AttnRes routing parameters, so every pass routes
separately. The looped presets load by name in training, evaluation and the
server (`--model_size`).

The layout of the trunk and its cooldown branches, the in-loop evals and the binding
batteries are described in
[docs/USAGE.md](docs/USAGE.md#training-runs-in-loop-evals-and-the-binding-batteries).

## Citation

```bibtex
@misc{kotodama_tsumugi_2026,
  title        = {kotodama-tsumugi-1b: a 1.4B looped base model with block attention residuals},
  author       = {Luxia and {Anima Labs}},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/LuxiaSL/kotodama-tsumugi-1b}},
  note         = {Code: https://github.com/anima-research/kotodama}
}
```
