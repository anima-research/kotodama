"""Extending a calibration's positional means past its reach, under the base's basis.

A loom that reads long documents reads positions the calibration never filled, and a position
past the last filled row is uncorrected: its residual-source features are computed against a
zero mean under the corrected names. Refitting the calibration over longer texts would refit the
basis too, which makes it a different calibration and every signature made under the old one
incomparable. Appending keeps what the base fixed and adds only what it lacks:

* the basis file is the base's, byte for byte;
* the means and position counts at positions ``0 .. base_reach - 1`` are the base's, byte for
  byte; only positions from ``base_reach`` on are new;
* the new rows come from long streams replayed and reduced in exactly the base's arithmetic,
  which is shown, not assumed: the base's own token streams, replayed by the same path, must
  rebuild the base means byte for byte before any tail is written.

The tail is a splice, so its seam is measured: the new means on the rows just under the base
reach against the base's rows, beside the new means' own split-half noise, and the corrected
norm ``|h_p - m_p|`` across the reach on a few long streams.

The pieces here are numeric and model-free; :mod:`kotodama.eval.anamnesis_calibrate` drives them
(``--append_from``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

#: A position whose mean averages this many streams or fewer is left at zero in the tail. Higher
#: than anamnesis's own fit floor, because a loom reads the tail of every long document.
DEFAULT_TAIL_COUNT_FLOOR = 50

#: Rows on each side of the base reach the seam check reads.
SEAM_SPAN = 64


class CalibrationAppendError(ValueError):
    """An append that would not keep the base's rows, basis or arithmetic."""


def means_from(sums: NDArray[np.float64], counts: NDArray[np.int64], floor: int) -> NDArray[np.float32]:
    """Per-position means (``sums`` is ``[layer, position, unit]``), a position at or under
    ``floor`` states left at zero. The arithmetic of anamnesis's fit: the float64 quotient,
    then float32."""
    covered = counts > floor
    safe = np.where(covered, counts, 1)
    out = (sums / safe[None, :, None]).astype(np.float32)
    out[:, ~covered] = 0.0
    return out


def splice(base_means: NDArray[np.float32], base_counts: NDArray[np.int64] | None, base_reach: int,
           new_means: NDArray[np.float32], new_counts: NDArray[np.int64],
           floor: int) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """(means, ``pos_counts``) whose rows ``0 .. base_reach - 1`` are the base's verbatim and
    whose tail is the new fit's, with a tail row at or under ``floor`` zeroed and counted 0."""
    if base_means.ndim != 3 or new_means.ndim != 3:
        raise CalibrationAppendError("means are indexed [layer, position, unit]")
    if base_means.shape[0] != new_means.shape[0] or base_means.shape[2] != new_means.shape[2]:
        raise CalibrationAppendError(
            f"base means {base_means.shape} and new means {new_means.shape} differ in layers or width")
    if base_means.dtype != new_means.dtype:
        raise CalibrationAppendError(f"base dtype {base_means.dtype} != new {new_means.dtype}")
    width = new_means.shape[1]
    if width <= base_reach:
        raise CalibrationAppendError(f"the new table ({width} positions) reaches no further than the "
                                     f"base ({base_reach})")
    depth = new_means.shape[0]
    means = new_means.copy()
    means[:, :base_reach] = base_means[:, :base_reach]
    thin = new_counts <= floor
    thin[:base_reach] = False
    means[:, thin] = 0.0
    counts = np.broadcast_to(new_counts[None, :], (depth, width)).astype(np.int64).copy()
    counts[:, thin] = 0
    if base_counts is not None:
        counts[:, :base_reach] = base_counts[:, :base_reach]
    if not np.array_equal(means[:, :base_reach], base_means[:, :base_reach]):
        raise CalibrationAppendError("the spliced head is not the base's rows byte for byte")
    return means, counts


def _rel(a: NDArray[Any], b: NDArray[Any]) -> NDArray[np.float64]:
    return (np.linalg.norm(a - b, axis=-1)
            / np.maximum(np.linalg.norm(b, axis=-1), 1e-12)).astype(np.float64)


def _q(values: NDArray[Any]) -> dict[str, float]:
    v = np.asarray(values, np.float64)
    return {"median": float(np.median(v)), "p95": float(np.percentile(v, 95)), "max": float(v.max())}


def seam_report(base_means: NDArray[np.float32], base_reach: int, new_means: NDArray[np.float32],
                even: NDArray[np.float32], odd: NDArray[np.float32], spliced: NDArray[np.float32],
                probes: dict[int, NDArray[np.float32]], probe_start: int,
                span: int = SEAM_SPAN) -> dict[str, Any]:
    """How the splice sits. ``new_vs_base_rel`` is the relative distance of the new means from
    the base's on the ``span`` rows under the reach; ``split_half_noise_rel`` is the new means'
    own noise on those rows (even vs odd streams, scaled by 1/sqrt 2), the scale the first is
    read against. ``corrected_norm_steps`` is, per probe stream, the largest step in the mean
    corrected norm across the reach (8 rows each side), in units of its spread over the
    ``span`` rows before it, maximised over layers. ``probes[i]`` holds a stream's raw states
    ``[layer, rows]`` from ``probe_start``."""
    lo = max(0, base_reach - span)
    rows = slice(lo, base_reach)
    vs_base = _rel(new_means[:, rows], base_means[:, rows])
    noise = _rel(even[:, rows], odd[:, rows]) / np.sqrt(2.0)
    steps = []
    for stream, raw in sorted(probes.items()):
        p = base_reach - probe_start
        if not 0 < p < raw.shape[1]:
            continue
        cn = np.linalg.norm(raw - spliced[:, probe_start:probe_start + raw.shape[1]], axis=-1)
        before = cn[:, max(0, p - 8):p].mean(axis=1)
        after = cn[:, p:p + 8].mean(axis=1)
        spread = cn[:, max(0, p - span):p].std(axis=1) + 1e-12
        steps.append({"stream": int(stream),
                      "max_abs_step_in_sd": float(np.max(np.abs(after - before) / spread))})
    return {"rows": [lo, base_reach - 1], "new_vs_base_rel": _q(vs_base),
            "split_half_noise_rel": _q(noise),
            "ratio_median": float(np.median(vs_base) / max(float(np.median(noise)), 1e-12)),
            "corrected_norm_steps": steps}
