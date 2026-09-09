"""Codex review point: "Test 'wrong-layout pairing' deliberately. A
native-layout buffer reaching the legacy kernel — or vice versa — should be
structurally impossible or caught immediately."

This is pure Python, no GPU/CUDA needed — it tests the *design*, not the
kernel. Run directly:

    python3 dev_probe/test_wrong_layout_pairing.py

The structural argument: as long as every one of repack / kernel
specialization / dispatch calls resolve_weight_smem_layout() with the same
inputs (see layout_identity.py) and threads the *result* forward instead of
each re-deriving their own boolean, a mismatch is structurally impossible —
there's only one computation to get right, not four to keep in sync.

This file exercises the failure mode Codex is actually worried about: what
happens if something upstream *does* end up passing a layout identity that
doesn't match what the current hardware/toolkit could have produced. That
can legitimately happen once repacked weights are ever cached across
process runs (flagged as a dormant risk in the blog draft on this branch,
same risk vLLM's PR flagged for itself) — a weight repacked on an SM90 box
with CUDA 13.4, then loaded by a process on an older toolkit or a
non-SM90 GPU. The loader must refuse that combination loudly, not silently
misinterpret the bytes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from layout_identity import WeightSmemLayout, ldmatrix_s4_available  # noqa: E402


class LayoutMismatchError(RuntimeError):
    """Raised when a tensor's recorded layout identity isn't achievable on
    the current (toolkit, GPU) — i.e. it could only have been produced by a
    different environment than the one now trying to consume it."""


def assert_layout_consumable(
    tensor_layout: WeightSmemLayout,
    *,
    sm_version: int,
    nvcc_version: tuple[int, int],
) -> None:
    """The loader-side guard: before dispatching to a kernel variant, check
    that the layout the tensor actually carries is one the current
    environment could have produced. This is the "caught immediately"
    half of Codex's requirement — structural impossibility isn't available
    here (a raw tensor is just bytes, it can't self-describe), so this
    explicit check is the enforcement mechanism instead.
    """
    if tensor_layout == WeightSmemLayout.WGMMA_A_LDMATRIX_S4:
        if not ldmatrix_s4_available(sm_version, nvcc_version):
            raise LayoutMismatchError(
                f"tensor was repacked as {tensor_layout.value}, which requires "
                f"CUDA>=13.4 on an SM90-family GPU, but the current environment "
                f"is sm_{sm_version} / nvcc {nvcc_version}. Refusing to dispatch "
                f"the legacy dequant kernel against ldmatrix.s8.s4-packed bytes "
                f"(or vice versa) — that would silently compute wrong numbers, "
                f"not crash."
            )


def test_native_layout_on_capable_hardware_is_accepted():
    assert_layout_consumable(
        WeightSmemLayout.WGMMA_A_LDMATRIX_S4, sm_version=90, nvcc_version=(13, 4)
    )
    print("[ok] native layout accepted on SM90 / CUDA 13.4")


def test_legacy_layout_is_always_accepted():
    # The legacy layout has no hardware/toolkit precondition — it's the
    # universal fallback, so it must never be rejected by this guard.
    assert_layout_consumable(
        WeightSmemLayout.LEGACY_DEQUANT_INTERLEAVED, sm_version=80, nvcc_version=(12, 8)
    )
    assert_layout_consumable(
        WeightSmemLayout.LEGACY_DEQUANT_INTERLEAVED, sm_version=90, nvcc_version=(13, 4)
    )
    print("[ok] legacy layout accepted regardless of hardware/toolkit")


def test_native_layout_on_incapable_hardware_is_rejected_loudly():
    # Simulates: a weight was repacked elsewhere (SM90 + CUDA 13.4) and
    # then loaded on this box, which can't produce that layout itself.
    try:
        assert_layout_consumable(
            WeightSmemLayout.WGMMA_A_LDMATRIX_S4, sm_version=89, nvcc_version=(12, 8)
        )
    except LayoutMismatchError:
        print("[ok] mismatched layout on incapable hardware raised, as required")
        return
    raise AssertionError(
        "expected LayoutMismatchError for WGMMA_A_LDMATRIX_S4 on sm_89/cuda 12.8, "
        "got no exception — this is exactly the silent-wrong-numbers failure mode "
        "Codex's review is asking to be prevented"
    )


if __name__ == "__main__":
    test_native_layout_on_capable_hardware_is_accepted()
    test_legacy_layout_is_always_accepted()
    test_native_layout_on_incapable_hardware_is_rejected_loudly()
    print("all wrong-layout-pairing checks passed")
