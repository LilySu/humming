# Shared-memory contract: `ldmatrix.s8.s4`-sourced weight tile

Codex's review point: "Make the no-.trans consequence explicit in the
shared-memory contract: logical dimensions, physical strides, swizzle,
required alignment, and bank mapping." This document is that contract.
It's normative for anything reading `dev_probe/wgmma_pairing_check.cu` or
implementing the real loader — if a future change conflicts with something
stated here, this document is wrong and needs updating, not silently
overridden.

## Why this needs to be written down at all

`ldmatrix.sync.aligned.m8n16.x4.shared::cta.s8.s4` has no `.trans` variant.
Every other `ldmatrix` shape in common use (`.m8n8.x4.b16`, used elsewhere
in Humming for the activation operand) has both a transposed and
non-transposed form, so a kernel author can choose whichever storage order
is convenient and let `ldmatrix` do the transpose. `.s8.s4` doesn't offer
that choice. The weight tile's physical layout in shared memory **must
already be** the one `.s8.s4` expects — there's no load-time correction if
it isn't.

## The two operands this affects, and why they're treated differently

| | Humming's weight (raw-PTX register-`a`) | Humming's activation (raw-PTX descriptor-`b`) |
|---|---|---|
| Source mechanism | `ldmatrix.s8.s4`, per-warp, into registers | WGMMA smem descriptor, warpgroup-collective |
| Swizzle required | No — see below | Yes — Humming's existing `make_wgmma_smem_desc` swizzle applies, unchanged by this work |
| Storage order requirement | **K-major, mandatory** (no `.trans`) | Whatever the existing descriptor convention already requires (unchanged) |

Only the weight side is new here. The activation side is untouched — it
already goes through Humming's existing, working smem-descriptor
machinery, and this work does not change it.

## The weight-tile contract

- **Logical dimensions:** `M × K` where `M` is whatever WGMMA `M`-shape is
  in play (64 in the concrete instantiation checked by this probe:
  `wgmma.mma_async.sync.aligned.m64n64k32.s32.s8.s8`) and `K = 32` for an
  int8-typed multiply (`256 / 8`, matching the same formula Humming already
  uses in `humming/kernel/humming.py:265` for `mma_shape_k`).
- **Element storage:** each logical element is a packed signed 4-bit
  nibble, stored as `stored_value = v XOR 0x8` (the same repack transform
  discussed throughout this branch — see `dev_probe/layout_identity.py`
  and the repack-kernel patch plan). Two nibbles pack into one byte.
- **Physical stride: K-major, mandatory.** Each of the `M` rows occupies
  `K / 2` contiguous bytes (16 bytes for `K = 32`), with **no gap** between
  a row's last byte and the next row's first byte unless deliberately
  padded (see alignment note below). This is not a preference — it's the
  only storage order `ldmatrix.s8.s4` (no `.trans`) can consume correctly.
  If a future change needs the weight tile transposed relative to this,
  that transpose must happen at repack time (writing K-major bytes in the
  first place), not at load time.
- **Alignment:** each `m8n16` sub-tile's base address must be 16-byte
  aligned (`__align__(16)` on the shared buffer, as used throughout
  `dev_probe/ldmatrix_s4_layout.cu`) — this is a hardware requirement of
  `ldmatrix` in general, not specific to `.s8.s4`.
- **Bank mapping / swizzle: none required for this operand.** Because the
  weight tile is consumed by `ldmatrix` (per-thread computed address) and
  not by a WGMMA smem descriptor, none of WGMMA's descriptor-level swizzle
  logic (`make_wgmma_smem_desc`'s `swizzle_type`/`stride` encoding) applies
  to it. Ordinary shared-memory bank-conflict considerations for the
  32-lane addressing pattern still apply (each lane in a warp should
  address a distinct bank where possible), but this is the same class of
  concern as any other `ldmatrix`-fed loader in Humming already handles
  (e.g. `S2RMemoryLoaderA`), not something new introduced by `.s8.s4`.

## What "K-major" concretely rules out

A row-major-in-`M` **and** row-major-in-`K` layout (the contract above) is
fine. A layout that's row-major in `K` but tiled/blocked across `M` in some
other order (e.g., the way `use_packed_k_layout`'s *existing* repack output
already rearranges data — see `process.cuh`'s `kUsePackedKLayout` branch,
lines 284-307) is **not** automatically compatible just because it's
"K-major" in a loose sense — the exact byte-for-byte row boundary matters.
This is precisely why `dev_probe/wgmma_pairing_check.cu`'s A-tile packing
(`pack_a_k_major()`) constructs the tile directly, from scratch, rather
than reusing `use_packed_k_layout`'s current output format unmodified — the
two need to be reconciled explicitly as part of the real repack-kernel
patch (see the `kUseLdmatrixS4` XOR insertion point discussed earlier on
this branch), not assumed compatible.
