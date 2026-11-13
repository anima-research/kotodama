"""shape_cost.py --profile bookkeeping: the GPU-busy union of overlapping kernel intervals and the kernel families."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "benchmark" / "shape_cost.py"
_spec = importlib.util.spec_from_file_location("shape_cost", _PATH)
shape_cost = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(shape_cost)


def test_union_merges_overlaps_and_keeps_gaps():
    # [0,10) and [5,20) overlap -> 20 us; [30,40) is separate -> 10 us; gap 20..30 is idle.
    assert shape_cost._union_ms([(30.0, 40.0), (0.0, 10.0), (5.0, 20.0)]) == pytest.approx(0.030)
    assert shape_cost._union_ms([]) == 0.0


@pytest.mark.parametrize("name,fam", [
    ("nvjet_tst_192x192_64x3_2x1_v_bz_coopA_NNT", "gemm"),
    ("cutlass3x_sm100_tensorop_gemm_bf16", "gemm"),
    ("flash_fwd_kernel", "attention"),
    ("triton_red_fused_add_mul_rsqrt_0", "other"),
])
def test_kernel_family(name, fam):
    assert shape_cost._family(name) == fam
