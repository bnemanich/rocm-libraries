################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT
################################################################################
"""Tail-loop emit content assertions for the subtile any-K tail.

Drives `KernelWriter._emitTailLoopScaffoldSubtile` at ASEM ∈ {32, 8,
4, 2, 1} via `setdefault_tail_scaffold_kernel_keys(..., asem=...)` and
pins the K%32 / K%8 / K%4 / K%2 / odd-K emit shape.

When the sub-lane byte refine fires (ASEM<numMIInUnroll for non-MX
integer-bpe operands) it owns the per-lane K-tail mask end-to-end:
the coarse `kPos vs LoopCounterL` cmp + per-VGPR cndmask is removed
and a single per-(operand, ir) mask chain feeds a `v_and` against
each boundary VGPR. The mask chain folds in mod = `elementsPerVgpr`-1
down to 0 byte slots, statically skipped when `ASEM*bpe % bpr == 0`
and runtime-gated by `LoopCounterL & (elementsPerVgpr-1)` otherwise.
"""
import re

import pytest

from Tensile.Tests.unit._subtile_tailloop_fixtures import (
    build_minimal_subtile_kwa,
    setdefault_tail_scaffold_kernel_keys,
    wrap_with_skiptoend,
)
from Tensile.Tests.unit.test_subtile_tailloop_emit import (
    _create_kernel,
    _extract_tail_section,
)


# ── Driver ──────────────────────────────────────────────────────────────────

def _emit_anyk_tail_asm(*, asem: int, pgr: int = 0, MT0: int = 128, MT1: int = 128) -> str:
    """Emit a BF16 subtile tail scaffold at the requested ASEM."""
    kernel = _create_kernel(MT0=MT0, MT1=MT1, fp4=False,
                            depthU=64, no_tail_loop=False)
    setdefault_tail_scaffold_kernel_keys(kernel, pgr, asem=asem)

    kwa = build_minimal_subtile_kwa(kernel)
    tPA = {"is_sparse": False, "tpsMetadata": None}
    tPB = {"is_sparse": False, "tpsMetadata": None}
    module = kwa._emitTailLoopScaffoldSubtile(kernel, tPA, tPB)
    return wrap_with_skiptoend(module)


# ── Tests: K%8 (ASEM=8) ──────────────────────────────────────────────────────

class TestAnyKEmit_K8:
    """K%8 reuses the K32 emit path unchanged: at numMIInUnroll=8 for
    bf16 the byte refine gate (`ASEM<numMIInUnroll`) is not satisfied,
    so the coarse per-mmak cmp + per-VGPR cndmask is the only mask.
    """

    def test_k8_emit_matches_k32_shape(self):
        asm_k32 = _emit_anyk_tail_asm(asem=32, pgr=0)
        asm_k8 = _emit_anyk_tail_asm(asem=8, pgr=0)
        tail_k32 = _extract_tail_section(asm_k32)
        tail_k8 = _extract_tail_section(asm_k8)
        assert tail_k32, "K32 baseline emitted no tail block"
        assert tail_k8, "K8 emit produced no tail block"

        cmp_k32 = re.findall(r"v_cmp_ge_i32.*LoopCounterL", tail_k32)
        cmp_k8 = re.findall(r"v_cmp_ge_i32.*LoopCounterL", tail_k8)
        cnd_k32 = re.findall(r"v_cndmask_b32.*[vV]alu", tail_k32)
        cnd_k8 = re.findall(r"v_cndmask_b32.*[vV]alu", tail_k8)

        assert len(cmp_k8) == len(cmp_k32), (
            f"K%8 (ASEM=8) lane-mask cmp count {len(cmp_k8)} must "
            f"equal K%32 (ASEM=32) baseline {len(cmp_k32)}. K%8 is "
            f"expected to reuse the K32 emit path unchanged."
        )
        assert len(cnd_k8) == len(cnd_k32), (
            f"K%8 (ASEM=8) lane-mask cndmask count {len(cnd_k8)} must "
            f"equal K%32 (ASEM=32) baseline {len(cnd_k32)}."
        )

    def test_k8_no_byte_refine(self):
        """ASEM=8 = numMIInUnroll → byte-refine gate must reject; no
        `byteRefine` comments should appear.
        """
        tail = _extract_tail_section(_emit_anyk_tail_asm(asem=8, pgr=0))
        assert tail
        assert "byteRefine" not in tail, (
            "K%8 emit must NOT engage the sub-lane byte refine "
            "(ASEM>=numMIInUnroll → coarse mask suffices)."
        )


# ── Tests: K%4 (ASEM=4) ──────────────────────────────────────────────────────

class TestAnyKEmit_K4:
    """ASEM=4 fires the byte refine (4<numMIInUnroll=8) but the partial
    mod>0 chain collapses statically because `ASEM*bpe = 8` is a
    multiple of `bpr = 4`. Only the mod=0 step is emitted; no runtime
    gate, no mod>0 mask byte, and the coarse cmp+cndmask is gone (#5).
    """

    def test_k4_byte_refine_mod0_only(self):
        tail = _extract_tail_section(_emit_anyk_tail_asm(asem=4, pgr=0))
        assert tail, "K%4 emit produced no tail block"

        # Byte refine must fire.
        assert "byteRefine" in tail, (
            "K%4 (ASEM=4) emit must engage the sub-lane byte refine "
            "(ASEM<numMIInUnroll)."
        )
        # Partial-mod chain is statically skipped: no mod>0 mask byte
        # mov, no partial-skip label.
        assert re.search(
            r"v_mov_b32\s+v\d+,\s*0xffff\b[^\n]*keep mask",
            tail, re.IGNORECASE,
        ) is None, (
            "K%4 emit must NOT contain a mod>0 keep-mask `0xFFFF` "
            "mov: ASEM*bpe (=8) is a multiple of bpr (=4), so the "
            "partial chain collapses statically."
        )
        assert "SubtileTailByteShiftPartialSkip" not in tail, (
            "K%4 emit must NOT contain the partial-mod skip label "
            "(static skip should drop it)."
        )
        assert re.search(
            r"s_and_b32[^\n]*LoopCounterL[^\n]*partial-mod residue",
            tail,
        ) is None, (
            "K%4 emit must NOT compute the runtime partial-mod "
            "residue (static skip should drop it)."
        )

        # mod=0 chain still fires per (operand, ir).
        seed_movs = re.findall(
            r"v_mov_b32\s+v\d+,\s*0xffffffff[^\n]*mask seed = full keep",
            tail, re.IGNORECASE,
        )
        assert len(seed_movs) >= 4, (
            "K%4 emit missing per-(operand, ir) mask seed "
            "`v_mov_b32 vMask, 0xFFFFFFFF`. Tail excerpt:\n"
            + tail[:1500]
        )
        # No coarse `kPosCur = kPosBase + mmak * miK` cmp/cndmask
        # when byte refine fires (#5).
        assert re.search(
            r"v_cndmask_b32[^\n]*if K_idx >= sizeL", tail
        ) is None, (
            "K%4 emit must NOT contain the legacy coarse "
            "`v_cndmask_b32 ... if K_idx >= sizeL` per-VGPR cndmask "
            "(byte refine subsumes it; #5)."
        )


# ── Tests: K%2 (ASEM=2) ──────────────────────────────────────────────────────

class TestAnyKEmit_K2:
    """ASEM=2 fires the byte refine. `ASEM*bpe = 4` is a multiple of
    `bpr = 4`, so the partial mod>0 chain is statically dropped: only
    the mod=0 step is emitted per (operand, ir). The coarse cmp +
    per-VGPR cndmask is removed (#5); the byte refine's mod=0 v_and
    against each boundary VGPR is the sole K-tail mask.
    """

    def test_k2_emits_per_operand_byte_refine(self):
        asm_k32 = _emit_anyk_tail_asm(asem=32, pgr=0)
        asm_k2 = _emit_anyk_tail_asm(asem=2, pgr=0)
        tail_k2 = _extract_tail_section(asm_k2)
        tail_k32 = _extract_tail_section(asm_k32)
        assert tail_k2, "K%2 emit produced no tail block"
        assert tail_k32, "K%32 baseline emit produced no tail block"

        # Byte refine emits one mask chain per (operand, ir): each
        # chain starts with `v_mov vMask, 0xFFFFFFFF` then folds the
        # mod=0 cndmask + per-VGPR v_and. Strictly more cmps than
        # the K%32 baseline (which only has the per-mmak coarse cmp).
        cmp_k2 = re.findall(r"v_cmp_ge_i32.*LoopCounterL", tail_k2)
        cmp_k32 = re.findall(r"v_cmp_ge_i32.*LoopCounterL", tail_k32)
        assert len(cmp_k2) > len(cmp_k32), (
            f"K%2 emit must add per-(operand, ir) cmps beyond the "
            f"K%32 baseline; got cmp_k2={len(cmp_k2)} vs "
            f"cmp_k32={len(cmp_k32)}."
        )

        # Per-(operand, ir) mask seed: `v_mov_b32 vMask, 0xFFFFFFFF`
        # tagged with the seed comment.
        seed_movs = re.findall(
            r"v_mov_b32\s+v\d+,\s*0xffffffff[^\n]*mask seed = full keep",
            tail_k2, re.IGNORECASE,
        )
        assert len(seed_movs) >= 4, (
            "K%2 emit missing per-(operand, ir) mask seed "
            "`v_mov_b32 vMask, 0xFFFFFFFF`. Tail excerpt:\n"
            + tail_k2[:1500]
        )

        # mod=0 cndmask: src1=0 (inline) folds vMask → 0 on past-
        # boundary lanes, leaves it at the prior chain value
        # otherwise. Tag pinned to the comment so future commenting
        # changes get caught.
        mod0_cnd = re.findall(
            r"v_cndmask_b32\s+v(\d+),\s+v\1,\s*0,\s+s\[\d+:\d+\]"
            r"[^\n]*mod=0[^\n]*past \? 0 : prev",
            tail_k2,
        )
        assert mod0_cnd, (
            "K%2 emit missing per-(operand, ir) mod=0 cndmask "
            "`v_cndmask_b32 vMask, vMask, 0, s[<lo>:<hi>]`. Tail "
            "excerpt:\n" + tail_k2[:2000]
        )

        # Per-VGPR mask application: `v_and_b32 vIdx, vMask, vIdx`.
        and_a = re.search(
            r"v_and_b32\s+v(\d+),\s+v\d+,\s+v\1"
            r"[^\n]*apply mask to ValuA",
            tail_k2,
        )
        and_b = re.search(
            r"v_and_b32\s+v(\d+),\s+v\d+,\s+v\1"
            r"[^\n]*apply mask to ValuB",
            tail_k2,
        )
        assert and_a is not None, (
            "K%2 emit missing per-VGPR `v_and_b32 vIdx, vMask, vIdx` "
            "for ValuA boundary VGPR. Tail excerpt:\n" + tail_k2[:2000]
        )
        assert and_b is not None, (
            "K%2 emit missing per-VGPR `v_and_b32 vIdx, vMask, vIdx` "
            "for ValuB boundary VGPR. Tail excerpt:\n" + tail_k2[:2000]
        )

        # ir-th VGPR's K_pos offset must appear in the v_add chain.
        # For elementsPerVgpr=2 (bf16) and vgprPerInUnroll=4 the
        # mmak=0 mod=0 offsets are {0, 2, 4, 6}; mmak=1 offsets
        # are {32, 34, 36, 38}. Pin both ir=1 mmak=0 (=2) and
        # ir=1 mmak=1 (=34) so a future change that drops the
        # per-mmak K_pos offset is caught.
        kpos_offsets = re.findall(
            r"v_add_u32\s+v\d+,\s+(\d+),\s+v\d+\s*//\s*byteRefine",
            tail_k2,
        )
        observed_offsets = {int(x) for x in kpos_offsets}
        assert 2 in observed_offsets, (
            f"K%2 byte refine missing ir=1 mmak=0 mod=0 K_pos "
            f"offset (=2). Observed offsets: "
            f"{sorted(observed_offsets)}. Tail excerpt:\n"
            f"{tail_k2[:1500]}"
        )
        assert 34 in observed_offsets, (
            f"K%2 byte refine missing ir=1 mmak=1 mod=0 K_pos "
            f"offset (=34 = 1*32 + 1*2). Observed offsets: "
            f"{sorted(observed_offsets)}. Tail excerpt:\n"
            f"{tail_k2[:1500]}"
        )

        # Negative: byte-shift refinement must NOT use a 64-bit lshl
        # or compute K_remain mod 8.
        assert re.search(r"v_lsh(?:l|lrev)_b64", tail_k2) is None, (
            "K%2 emit must NOT contain v_lshl(rev)_b64."
        )
        assert re.search(r"s_and_b32[^\n]*\b7\b", tail_k2) is None, (
            "K%2 emit must NOT compute `K_remain & 7`."
        )

        # Static partial-mod skip: ASEM=2 → ASEM*bpe=4 = bpr, so the
        # mod>0 chain (and its skip label, runtime gate, hi mask
        # seed) must NOT be emitted.
        assert "SubtileTailByteShiftPartialSkip" not in tail_k2, (
            "ASEM=2 emit must NOT contain the runtime partial-mod "
            "skip label (static skip should drop it)."
        )
        assert re.search(
            r"v_mov_b32\s+v\d+,\s*0xffff\b[^\n]*keep mask",
            tail_k2, re.IGNORECASE,
        ) is None, (
            "ASEM=2 emit must NOT contain the mod=1 `0xFFFF` keep-"
            "mask mov (static skip should drop it)."
        )

        # #5 negative pin: the legacy coarse per-VGPR cndmask is
        # subsumed by the byte refine's v_and when the refine fires.
        assert re.search(
            r"v_cndmask_b32[^\n]*if K_idx >= sizeL", tail_k2
        ) is None, (
            "ASEM=2 emit must NOT contain the legacy coarse "
            "`v_cndmask_b32 ... if K_idx >= sizeL` per-VGPR cndmask "
            "(byte refine subsumes it; #5)."
        )


# ── Tests: K%1 (odd K, ASEM=1) ───────────────────────────────────────────────

class TestAnyKEmit_K1:
    """ASEM=1 (odd K). `ASEM*bpe = 2` is not a multiple of `bpr = 4`,
    so the mod>0 partial chain is emitted under a runtime gate
    `s_and ..., LoopCounterL, (elementsPerVgpr-1)` per (operand, ir)
    chain. The chain shape is:

      v_mov_b32 vMask, 0xFFFFFFFF                         // mask seed
      s_and_b32 sGate, sgprLoopCounterL, (elementsPerVgpr-1)
      s_cmp_eq_u32 sGate, 0
      s_cbranch_scc1 SubtileTailByteShiftPartialSkip_*
      v_mov_b32 vSeed, 0xFFFF                             // mod=1 keep
      v_add_u32 vKpos, mod=1 offset, kPosBase
      v_cmp_ge_i32 sMask, vKpos, sgprLoopCounterL
      v_cndmask_b32 vMask, vMask, vSeed, sMask
    SubtileTailByteShiftPartialSkip_*:
      v_add_u32 vKpos, mod=0 offset, kPosBase
      v_cmp_ge_i32 sMask, vKpos, sgprLoopCounterL
      v_cndmask_b32 vMask, vMask, 0, sMask
      v_and_b32 vIdx, vMask, vIdx                         // each idx
    """

    def test_k1_no_narrow_load_and_emits_partial_chain(self):
        asm = _emit_anyk_tail_asm(asem=1, pgr=0)
        tail = _extract_tail_section(asm)
        assert tail, "ASEM=1 emit produced no tail block"

        # Narrow d16 load must not be emitted (illegal on gfx950).
        assert re.search(
            r"buffer_load_d16_b16|buffer_load_short_d16",
            asm,
        ) is None, (
            "ASEM=1 emit must NOT contain the narrow DTL load: "
            "`buffer_load_*_d16 ... lds` is not legal on gfx950."
        )

        # Runtime gate skip label.
        assert re.search(
            r"label_SubtileTailByteShiftPartialSkip", tail
        ), (
            "ASEM=1 emit missing partial-mod runtime gate skip "
            "label `SubtileTailByteShiftPartialSkip*`."
        )
        # Runtime gate `s_and ..., LoopCounterL, 0x1` (elementsPerVgpr=2).
        gate_match = re.search(
            r"s_and_b32[^\n]*LoopCounterL[^\n]*0x1"
            r"[^\n]*partial-mod residue",
            tail,
        )
        assert gate_match is not None, (
            "ASEM=1 emit missing `s_and_b32 sGate, "
            "sgprLoopCounterL, 0x1` partial-mod residue gate. "
            "Tail excerpt:\n" + tail[:2000]
        )

        # Per-mod=1 keep-mask seed `v_mov_b32 vSeed, 0xFFFF`.
        keep_mod1 = re.findall(
            r"v_mov_b32\s+v\d+,\s*0xffff\b[^\n]*mod=1[^\n]*keep mask",
            tail, re.IGNORECASE,
        )
        assert len(keep_mod1) >= 1, (
            "ASEM=1 emit missing mod=1 keep-mask `v_mov_b32 vSeed, "
            "0xFFFF`. Tail excerpt:\n" + tail[:2000]
        )

        # mod=1 cndmask: `vMask = past ? vSeed : prev` →
        # `v_cndmask_b32 vMask, vMask, vSeed, sMask`.
        mod1_cnd = re.search(
            r"v_cndmask_b32\s+v(\d+),\s+v\1,\s+v\d+,\s+s\[\d+:\d+\]"
            r"[^\n]*mod=1[^\n]*past \? 0xFFFF : prev",
            tail,
        )
        assert mod1_cnd is not None, (
            "ASEM=1 emit missing mod=1 cndmask `v_cndmask_b32 "
            "vMask, vMask, vSeed, s[<lo>:<hi>]` (the past-mod=1 "
            "selector). Tail excerpt:\n" + tail[:2000]
        )

        # mod=0 cndmask: `vMask = past ? 0 : prev`.
        mod0_cnd = re.search(
            r"v_cndmask_b32\s+v(\d+),\s+v\1,\s*0,\s+s\[\d+:\d+\]"
            r"[^\n]*mod=0[^\n]*past \? 0 : prev",
            tail,
        )
        assert mod0_cnd is not None, (
            "ASEM=1 emit missing mod=0 cndmask `v_cndmask_b32 "
            "vMask, vMask, 0, s[<lo>:<hi>]`. Tail excerpt:\n"
            + tail[:2000]
        )

        # Per-VGPR mask application for both ValuA and ValuB.
        and_a = re.search(
            r"v_and_b32\s+v(\d+),\s+v\d+,\s+v\1"
            r"[^\n]*apply mask to ValuA",
            tail,
        )
        and_b = re.search(
            r"v_and_b32\s+v(\d+),\s+v\d+,\s+v\1"
            r"[^\n]*apply mask to ValuB",
            tail,
        )
        assert and_a is not None, (
            "ASEM=1 emit missing `v_and_b32 vIdx, vMask, vIdx` "
            "(apply mask to ValuA[..]) per-VGPR mask application. "
            "Tail excerpt:\n" + tail[:2000]
        )
        assert and_b is not None, (
            "ASEM=1 emit missing `v_and_b32 vIdx, vMask, vIdx` "
            "(apply mask to ValuB[..]) per-VGPR mask application. "
            "Tail excerpt:\n" + tail[:2000]
        )

        # #5 negative pin: the legacy coarse per-VGPR cndmask is
        # subsumed by the byte refine's v_and when the refine fires.
        assert re.search(
            r"v_cndmask_b32[^\n]*if K_idx >= sizeL", tail
        ) is None, (
            "ASEM=1 emit must NOT contain the legacy coarse "
            "`v_cndmask_b32 ... if K_idx >= sizeL` per-VGPR "
            "cndmask (byte refine subsumes it; #5)."
        )

        # Negative: legacy hi16 step (yesterday's intermediate
        # shape) must not be present.
        legacy_and_ffff = re.search(
            r"v_and_b32\s+v\d+,\s*0xffff,\s+v\d+"
            r"[^\n]*hi16 -> 0",
            tail, re.IGNORECASE,
        )
        assert legacy_and_ffff is None, (
            "ASEM=1 emit must NOT contain the legacy per-VGPR "
            "`v_and_b32 v<tmp>, 0xFFFF, v<idx>  // ... hi16 -> 0` "
            "step (replaced by per-(operand, ir) mask chain)."
        )
        legacy_cnd_hi_per_vgpr = re.search(
            r"v_cndmask_b32\s+v(\d+),\s+v\1,\s+v\d+,\s+s\[\d+:\d+\]"
            r"[^\n]*hi16 Valu",
            tail,
        )
        assert legacy_cnd_hi_per_vgpr is None, (
            "ASEM=1 emit must NOT contain the legacy per-VGPR "
            "`v_cndmask_b32 v<idx>, v<idx>, v<tmp>, s[..]` for hi16 "
            "clear (replaced by per-(operand, ir) mask chain)."
        )

        # Negative: the prior helper's `SubtileTailByteHi16Skip`
        # label is gone — its purpose is owned by the unified
        # `SubtileTailByteShiftPartialSkip` runtime gate.
        assert "SubtileTailByteHi16Skip" not in tail, (
            "ASEM=1 emit must NOT contain the legacy "
            "`SubtileTailByteHi16Skip` label (the unified mask "
            "chain uses `SubtileTailByteShiftPartialSkip` instead)."
        )


# ── Regression net: K%32 emit must stay identical ────────────────────────────

class TestAnyKEmit_K32Unchanged:
    """K%32 (ASEM=32) regression net: structural fingerprint (cmp /
    cndmask presence + absence of any-K helper opcodes) so the K32
    emit cannot silently drift when the any-K helpers change.
    """

    K32_FINGERPRINT_HAS_LANE_CMP = True
    K32_FINGERPRINT_HAS_CNDMASK_VALUA = True
    K32_FINGERPRINT_HAS_CNDMASK_VALUB = True
    K32_FINGERPRINT_NO_LSHL_B64 = True
    K32_FINGERPRINT_NO_D16_B16 = True
    K32_FINGERPRINT_NO_AND_7 = True

    def test_k32_unchanged_with_new_helpers(self):
        asm = _emit_anyk_tail_asm(asem=32, pgr=0)
        tail = _extract_tail_section(asm)
        assert tail, "K%32 emit produced no tail block"

        cmp_match = re.search(r"v_cmp_ge_i32.*LoopCounterL", tail)
        assert (cmp_match is not None) == self.K32_FINGERPRINT_HAS_LANE_CMP, (
            "K%32 emit lane cmp shape changed"
        )
        cnd_a = re.search(r"v_cndmask_b32.*[vV]aluA", tail)
        assert (cnd_a is not None) == self.K32_FINGERPRINT_HAS_CNDMASK_VALUA, (
            "K%32 emit cndmask valuA shape changed"
        )
        cnd_b = re.search(r"v_cndmask_b32.*[vV]aluB", tail)
        assert (cnd_b is not None) == self.K32_FINGERPRINT_HAS_CNDMASK_VALUB, (
            "K%32 emit cndmask valuB shape changed"
        )

        no_lshl = re.search(r"v_lsh(?:l|lrev)_b64", tail) is None
        assert no_lshl == self.K32_FINGERPRINT_NO_LSHL_B64, (
            "K%32 emit must NOT contain v_lshl(rev)_b64"
        )
        no_d16 = re.search(
            r"buffer_load_d16_b16|buffer_load_short_d16|BufferLoadD16B16",
            tail,
        ) is None
        assert no_d16 == self.K32_FINGERPRINT_NO_D16_B16, (
            "K%32 emit must NOT contain buffer_load_d16_b16"
        )
        no_and_7 = re.search(
            r"s_and_b32[^\n]*(?:\b7\b|\b8\s*-\s*1\b)", tail
        ) is None
        assert no_and_7 == self.K32_FINGERPRINT_NO_AND_7, (
            "K%32 emit must NOT compute K_remain mod 8 via "
            "`s_and_b32 ..., 7`"
        )


# ── Direct predicate test: `_subtileTailByteShiftApplies` ────────────────────

def _build_predicate_kernel(*, asem, mxa, mxb, bpeA, bpeB):
    """Minimal kernel-dict driving `_subtileTailByteShiftApplies`.

    The predicate only reads `AssertSummationElementMultiple`,
    `ProblemType.MXBlockA`, `ProblemType.MXBlockB`, and
    `ProblemType.DataType{A,B}.numBytes()`. No other state is needed.
    """
    class _Bpe:
        def __init__(self, n):
            self._n = n

        def numBytes(self):
            return self._n

    return {
        "AssertSummationElementMultiple": asem,
        "ProblemType": {
            "MXBlockA": mxa,
            "MXBlockB": mxb,
            "DataTypeA": _Bpe(bpeA),
            "DataTypeB": _Bpe(bpeB),
        },
    }


# bpeA / bpeB legend: bf16/fp16 = 2, mxfp8/int8 = 1, mxfp4 = 0.5,
# fp32 = 4. The relaxed gate accepts any operand pair with integer
# bpe in [1, bpr]; sub-byte (mxfp4: 0.5) and >register (>4) are out.
@pytest.mark.parametrize(
    "asem,numMIInUnroll,mxa,mxb,bpeA,bpeB,expected",
    [
        # Inside the asem<numMIInUnroll window, no MX, integer bpe.
        (4,  8, 0,  0, 2,   2,   True),    # bf16 / fp16 (homogeneous)
        (4,  8, 0,  0, 1,   1,   True),    # int8/fp8 (homogeneous; #3
                                            # gate now accepts integer bpe)
        (4,  8, 0,  0, 2,   1,   True),    # mixed bf16/fp8 (#3 enables
                                            # per-operand handling)
        (4,  8, 0,  0, 4,   4,   True),    # fp32 (1 element/VGPR);
                                            # mod=0-only chain is valid
        # MX path: predicate must reject regardless of asem/bpe.
        (4, 32, 32, 0, 1,   2,   False),   # mxfp8 A
        (4, 32, 0, 32, 2,   0.5, False),   # mxfp4 B
        # asem >= numMIInUnroll: coarse mask covers everything.
        (8,  8, 0,  0, 2,   2,   False),
        (32, 8, 0,  0, 2,   2,   False),
        # Sub-byte / non-integer bpe: helper assumes byte-aligned mod
        # boundaries, so mxfp4 (numBytes=0.5) must be rejected.
        (4,  8, 0,  0, 0.5, 0.5, False),
        (4,  8, 0,  0, 2,   0.5, False),   # mixed bf16 / mxfp4
    ],
)
def test_subtile_tail_byte_shift_applies_predicate(
    asem, numMIInUnroll, mxa, mxb, bpeA, bpeB, expected
):
    """Pin the predicate's truth table across asem / MX / dtype.

    The gate now accepts any non-MX operand pair with integer bpe in
    `[1, bpr]` so the helper can fire on fp8/int8/fp16/bf16/fp32; only
    the MX path and ASEM>=numMIInUnroll force a False.
    """
    from Tensile.KernelWriter import KernelWriter

    kernel = _build_predicate_kernel(
        asem=asem, mxa=mxa, mxb=mxb, bpeA=bpeA, bpeB=bpeB
    )
    actual = KernelWriter._subtileTailByteShiftApplies(kernel, numMIInUnroll)
    assert actual is expected, (
        f"Predicate(asem={asem}, numMIInUnroll={numMIInUnroll}, "
        f"mxa={mxa}, mxb={mxb}, bpeA={bpeA}, bpeB={bpeB}) returned "
        f"{actual}, expected {expected}."
    )
