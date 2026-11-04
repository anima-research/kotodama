"""Audit a source before (or after) filtering: what would the battery do to it, and why?

    python -m kotodama.data.curation.audit --in_dir <chunks> --source NAME --class web --out_dir <audit>
    python -m kotodama.data.curation.audit --jsonl <file.jsonl[.zst]> --source NAME --class prose --out_dir <audit>

One pass over the source counts every document and reservoir-samples ``--n`` of
them uniformly. Each sampled document gets the filter metrics of
``kotodama.data.curation.filters`` and the verdict its class would receive. The
audit writes three files: ``<source>.summary.json`` (verdict and reason rates,
metric quantiles), ``<source>.exemplars.json`` (up to ``--exemplars`` snippets
per drop or repair reason, for reading) and ``<source>.metrics.jsonl`` (one row
per sampled document). Running it on raw input before choosing a source class
shows which rules a class would trip; running it on ingested chunks shows what
survived.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

from kotodama.data.curation.chunks import Record, discover_chunks, iter_records
from kotodama.data.curation.filters import SourceClass, apply_filters

logger = logging.getLogger(__name__)

SNIPPET_CHARS = 1500
QUANTILE_KEYS = ("chars", "tags_per_10k", "entities_per_10k", "fffd_per_10k", "nonlatin_letter_frac",
                 "dup_line_frac", "top_bigram_frac")


def reservoir(records: Iterator[Record], n: int, seed: int) -> tuple[list[Record], int]:
    """A uniform sample of ``n`` records (Algorithm R) and the total count."""
    rng = random.Random(seed)
    sample: list[Record] = []
    total = 0
    for rec in records:
        if len(sample) < n:
            sample.append(rec)
        else:
            j = rng.randrange(total + 1)
            if j < n:
                sample[j] = rec
        total += 1
    return sample, total


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    v = sorted(values)

    def q(p: float) -> float:
        return v[min(int(p * len(v)), len(v) - 1)]

    return {"p50": q(0.5), "p90": q(0.9), "p99": q(0.99), "max": v[-1]}


def audit(records: Iterator[Record], source: str, source_class: SourceClass, out_dir: Path, n: int = 20_000,
          seed: int = 42, exemplars: int = 20) -> dict[str, Any]:
    """Sample, measure and judge; writes the three files and returns the summary."""
    sample, total = reservoir(records, n, seed)
    actions: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    for rec in sample:
        v = apply_filters(rec.text, source_class)
        m = asdict(v.metrics)
        rows.append({"line": rec.line, "action": v.action.value, "reasons": v.reasons, **m})
        actions[v.action.value] += 1
        for r in v.reasons:
            reasons[r] += 1
            ex = examples.setdefault(r, [])
            if len(ex) < exemplars:
                ex.append({"line": rec.line, "metrics": m, "snippet": rec.text[:SNIPPET_CHARS]})
    k = max(len(sample), 1)
    summary = {
        "source": source, "source_class": source_class.value, "total_docs": total, "sampled": len(sample),
        "action_pct": {a: round(100 * c / k, 2) for a, c in actions.items()},
        "reason_pct": {r: round(100 * c / k, 2) for r, c in reasons.most_common()},
        "quantiles": {key: _quantiles([row[key] for row in rows]) for key in QUANTILE_KEYS},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{source}.summary.json").write_text(json.dumps(summary, indent=1))
    (out_dir / f"{source}.exemplars.json").write_text(json.dumps(examples, indent=1, ensure_ascii=False))
    with open(out_dir / f"{source}.metrics.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [audit] %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="sample a source and report what the filters would do")
    ap.add_argument("--in_dir", type=Path, help="a chunk directory (with --source)")
    ap.add_argument("--jsonl", type=Path, help="one .jsonl / .jsonl.zst file of {text: ...} rows")
    ap.add_argument("--source", required=True)
    ap.add_argument("--class", dest="source_class", required=True, choices=[c.value for c in SourceClass])
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exemplars", type=int, default=20)
    args = ap.parse_args(argv)
    if (args.in_dir is None) == (args.jsonl is None):
        ap.error("give exactly one of --in_dir, --jsonl")
    if args.jsonl is not None:
        records: Iterator[Record] = iter_records(args.jsonl)
    else:
        chunks = discover_chunks(args.in_dir).get(args.source)
        if not chunks:
            print(f"error: no chunks of {args.source} in {args.in_dir}", file=sys.stderr)
            return 2
        records = (r for c in chunks for r in iter_records(c))
    summary = audit(records, args.source, SourceClass(args.source_class), args.out_dir, args.n, args.seed,
                    args.exemplars)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
