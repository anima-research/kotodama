# Contributing

Two kinds of contribution arrive here, and they are handled differently.

**Code** is a pull request against this repository. The gates described below decide whether it
merges.

**Claims** are not code. A contribution that asserts a finding — that a recipe improves a model,
that an architecture choice has an effect, that a number holds — arrives as an issue with the runs
and receipts behind it, is replicated where warranted, and is stated in a model card or a results
page only once it has been. Code that *enables* a claim (an eval, a probe, a training option) is
welcome in a pull request; the claim itself travels the other road.

## How this repository is organised

Each fact has one home, and every other place reads it from there:

- model shapes, AttnRes boundaries and the tokenizer: `src/kotodama/presets.py`;
- checkpoint loading and names: `src/kotodama/ckpt.py` with `configs/checkpoints.yaml`;
- ChatML and stop tokens: `src/kotodama/serve/chatml.py`; sampling laws: `src/kotodama/serve/laws.py`.

A constant restated as a literal somewhere else is a second source of truth, and the two drift.
Machine-specific paths, hostnames and keys never enter a tracked file: they live in `site.env`
(gitignored; `site.env.example` is the template), which `tools/_env.sh` loads. Training configs are
flat YAML whose keys are `train.py` flags, sharing blocks through `extends:`;
`tests/data/config_snapshot.json` pins every config's effective values and is regenerated only on
purpose (`KOTODAMA_WRITE_CONFIG_SNAPSHOT=1`). Capture and readout go through
`kotodama.model.capture`, which is parity-tested against `kotodama.model.llama`; a second AttnRes
forward mirror is refused. Every task has one golden path, listed in docs/USAGE.md; a change adds to
that path rather than beside it.

## The documentation rule

Documentation and comments obey two rules at once. Both are mechanically checked, and both apply
to three surfaces: comments, docstrings, and the message a `raise` or a logging call says out loud.
A stranger meets that last one at the moment something breaks, which is the worst moment to hand
them a pointer they cannot follow. The rules never apply to wire format — a fixture, a dict key, a
status value, a filename, a banked tag — which is content the program keys on, and which changing
would break reading what is already banked. They do apply to prose a run writes for a human: a
`note`, a `description` or a table caption in a report is read the way a comment is read, so it
states the standing fact and cites nothing. The checkers see the first three surfaces; the fourth
is a reviewer's catch. Inside an f-string, the substitutions are code and are read as such.

### 1. State what is true now

A comment describes the code as it stands. It does not narrate how the code got there.

No `previously`, `used to <verb>`, `changed in`, `we now`, `no longer`. No bare `TODO`, `FIXME` or
`HACK` — a known gap is a refusal the code makes explicitly, a test that pins the current
behaviour, or an issue, not a word left in a file for somebody to find.

**No dates.** Not on an edit, and not on evidence either. Git records when a line changed, and a
dated measurement or decision belongs in the record that holds it. What the code needs is the
standing fact it depends on, stated in the present tense.

```python
# Good — the constraint, and why it is load-bearing.
# The fp64 routing only exists inside a compiled graph: eager Triton launches
# type the eps fp32, so a lineage that trained compiled keeps its numerics only
# when the forward is compiled too.

# Bad — narrates an edit.
# We now compile by default; this used to run eager.

# Bad — dates the evidence instead of stating it.
# Loops survive repetition penalty 1.2 (felt read 2026-01-15, n=8).
```

### 2. Every referent must be reachable from this repository

If a comment or docstring names something, a reader holding only this repository must be able to
go look at it: a module path that exists here, a file in this tree, a symbol this package defines,
or a public URL.

It does **not** mean a path into a private tree, a planning document, a design note, an internal
ticket, or a conversation. It does not mean "see the earlier discussion" or "as decided
elsewhere". It does not mean a machine, a host, an account, a scheduler, or a sibling repository:
a reader has no login on the box that taught us something, and a comment crediting that box states
provenance where the constraint belongs — the memory ceiling, the shared-disk budget, the
bandwidth arithmetic, which hold wherever the code runs. That part is a reviewer's catch rather
than a checker's, by design: a public gate would have to enumerate the names, and publishing the
list defeats the point of keeping them out.

It also does not mean a **provenance citation** — a `§` of a document that is not here, a named arm
or milestone code, a pre-registration, a bare item code, a commit hash of this repository. The code
is the receipt for what the code does, and git is the receipt for how it came to do it. A pinned
commit of an *upstream* project is different: it is public, so it is written as a link.

When the substance is short, state it inline. When it is long and public, link it. When it is long
and private, restate the part this code depends on — one sentence of standing fact beats a citation
nobody can open.

```python
# Good — names a module in this repository.
# The span detector that labels KTO data is kotodama.posttrain.selfcopy.

# Good — the substance, stated, with nothing to chase.
# Reusing a name or an object is how a model tracks who has what, so the
# repetition labeller counts verbatim runs of the model's own text, never
# shared nouns.

# Bad — a referent this repository does not contain.
# See research/notes/threshold-memo for why the threshold is 0.2.

# Bad — cites the plan instead of stating the constraint.
# Gates per the release plan §3.
```

The rule has one consequence worth stating plainly: **a claim in a docstring must match the code
under it.** "This replaces X" is false while X is still running. Describe what is, and let what is
be the argument.

### Keep the prose style already here

Present tense. Say why, not only what — the constraint a reader would otherwise violate, the
failure a refusal exists to prevent. Document a refusal where the refusal happens. A module
docstring carries the shape of the module; a function's docstring carries what a caller needs:
arguments, units, the shape of what comes back, and how it fails.

## Pages a stranger reads

The README, model cards, results pages and the binding atlas are read by a technically literate
stranger who has never seen the research and may read only one page. They follow a stricter voice:

- **Third person, always.** No "we" or "our": the model, the run, the battery is the subject ("the
  run trained on 100B tokens", "the battery measured…").
- **Timeless present for standing facts, simple past for events.** "Looped depth reuses the core's
  weights" (standing); "the trunk trained to 100B tokens" (event). No futures, no roadmaps, no
  "currently", no "recently".
- **No self-narration.** A page never mentions its own edits or status. If a clause was wrong, the
  page states the correct clause; the reader does not pass through the error on the way to the
  correction.
- **Uncertainty is a fact about evidence**, not a mood: "measured on one seed", "not replicated at
  another scale". Never "we believe" or "it seems".
- **A claim travels with its scope**, in the same sentence or the next — one trunk seed, one shape,
  the harness it was measured with — written as part of the claim rather than a trailing
  limitations block.
- **Numbers are verbatim from the run that produced them, with n and the comparison inline**, every
  time they appear. Comparisons with other models are same-harness or say they are not.
- **Legible on one read.** A subject sits near its verb; an object relative keeps its "that" ("the tokens
  that its output favours"); at most two nouns stack before a head; the subject of a mind verb (decide,
  see, want) is a person, not a thing. `tools/check_legibility.py` reports these features against a
  Wikipedia reference and lists the sentences behind each count. It is a report, not a gate.
- **Plain and declarative.** No marketing and no mystique. Terms of art (AttnRes, looped depth,
  binding) are defined or linked at first use on each page.
- **Never on a public page:** local paths, machine or host names, account details, private
  throughput of private machines, names of unrelated private projects.

## The gates

A change merges when these pass. Each is a command you can run from the repository root.

| gate | what it checks | command |
|---|---|---|
| tests | the CPU suite, across all cores (pytest-xdist, from the `dev` extra; `-n 0` runs it serially) | `.venv/bin/python -m pytest tests/` |
| G2 test retention | a change that shrinks the suite shrinks the code by at least as many lines | `python -m tools.check_test_retention --base origin/main` |
| G3 documentation | the two rules above | `python -m tools.check_timelessness --root src/kotodama scripts tools tests` and `python -m tools.check_referents --root src/kotodama scripts tools tests` |
| G4 import closure | every module reachable from a script, a tool or a test; no orphans | `python -m tools.check_import_closure --package src/kotodama --roots scripts tests tools` |
| parity | a change to model, serve or train code ships with a test or a parity check | the test beside it; GPU parity scripts live in `scripts/utils/` and `scripts/benchmark/` |

G2 counts physical lines of Python under `tests/` (the suite) and everywhere else (the code), at
the branch and at the point it left `main`. Adding tests, adding code, or deleting a module together
with its tests all pass; a suite that shrinks while the code it covered stays fails.

Each G3 rule is pinned by a corpus rather than by reading: `tests/test_gate_fixtures.py` holds the
strings each checker must catch and the legitimate prose it must stay quiet on. A rule that stops
seeing something fails a test there instead of quietly reporting PASS; closing a blind spot starts
by adding the string that slipped through. Both pattern files — `tools/timelessness_allowlist.txt`
and `tools/referents_allowlist.txt` — carry a reason above every entry, and an entry is a claim that
its reason is true *now*. When a reason expires, the entry goes.

G4 reads `src/kotodama` statically. A module reached only from a shell launcher in `tools/` is an
orphan to it, which is the cue to give that module a test.

The GPU checks — decode-versus-prefill parity, lm-eval equivalence against a banked result, kernel
parity — cannot run on a CPU-only machine; they run where the hardware is and their receipt is
attached to the change. A gate may be waived only in writing, with the reason beside the receipt:
the failure is recorded, never the gate rewritten.

`python -m tools.surface_report --root src/kotodama` reports the size of the codebase in code
tokens and documentation words. It is a trend instrument, not a gate.

## Publishing

This repository is public. Before a push to `main`, the gates above pass and the diff is read for
the things no checker can enumerate: host and machine names, private paths, keys and tokens,
account names, and pointers into private research.
