"""Unit tests for routing quick-reduce payloads to AITER's FlyDSL INT4 kernel.

A payload FlyDSL cannot take is not an error: it falls back to the HIP
QuickReduce kernels, which looks exactly like FlyDSL being slow. These tests
pin down which payloads go where, and when the FlyDSL communicator is built.
"""

import sys
import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.distributed.device_communicators import quick_all_reduce as qr_mod
from sglang.srt.distributed.device_communicators.quick_all_reduce import (
    QuickAllReduce,
    QuickReduceRegime,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MB = 1024 * 1024


class _FakeFlyDSLInt4:
    instances = []

    def __init__(self, *, group=None, device=None, rank=0, world_size=8):
        self.init_kwargs = dict(
            group=group, device=device, rank=rank, world_size=world_size
        )
        self.compiled = []
        self.allreduce_calls = 0
        self.closed = False
        _FakeFlyDSLInt4.instances.append(self)

    def compile(self, inp, out):
        self.compiled.append((inp.dtype, inp.numel()))

    def allreduce(self, inp, out):
        self.allreduce_calls += 1

    def close(self):
        self.closed = True


def _make_quick_reduce(quant_level=QuickReduceRegime.INT4, world_size=8):
    """Build an enabled communicator without a process group or GPU.

    The constructor needs both; the routing under test needs neither.
    """
    quick_reduce = QuickAllReduce.__new__(QuickAllReduce)
    quick_reduce.disabled = False
    quick_reduce.group = object()
    quick_reduce.device = torch.device("cpu")
    quick_reduce.rank = 0
    quick_reduce.world_size = world_size
    quick_reduce.qr_quant_level = quant_level
    quick_reduce.use_fp16_kernels = 1
    quick_reduce.qr_max_size = 2048 * MB
    quick_reduce._ptr = 0
    return quick_reduce


def _bf16(num_elems):
    return torch.empty(num_elems, dtype=torch.bfloat16)


class TestFlyDSLInt4Dispatch(CustomTestCase):
    def _dispatch(self, inp, out=None):
        quick_reduce = _make_quick_reduce()
        fake = _FakeFlyDSLInt4()
        quick_reduce._flydsl_int4 = fake
        hip_calls = []
        with patch.object(
            qr_mod.ops,
            "qr_all_reduce",
            lambda *args: hip_calls.append(args),
            create=True,
        ):
            quick_reduce.quick_all_reduce(inp, out=out)
        return fake.allreduce_calls, len(hip_calls)

    def test_bf16_goes_to_flydsl(self):
        self.assertEqual(self._dispatch(_bf16(MB)), (1, 0))

    def test_fp16_stays_on_hip(self):
        inp = torch.empty(MB, dtype=torch.float16)
        self.assertEqual(self._dispatch(inp), (0, 1))

    def test_weakly_contiguous_stays_on_hip(self):
        # A transposed view passes is_weak_contiguous (the quick-reduce gate)
        # but not is_contiguous, which FlyDSL requires.
        inp = _bf16(2 * MB).view(1024, 2048).t()
        self.assertEqual(self._dispatch(inp), (0, 1))

    def test_misaligned_input_stays_on_hip(self):
        # Right size, but the data pointer is 8 bytes past a 16-byte boundary.
        inp = _bf16(MB + 16)[4 : MB + 4]
        self.assertEqual(inp.data_ptr() % 16, 8)
        self.assertEqual(self._dispatch(inp), (0, 1))

    def test_misaligned_output_stays_on_hip(self):
        out = _bf16(MB + 16)[4 : MB + 4]
        self.assertEqual(self._dispatch(_bf16(MB), out=out), (0, 1))

    def test_no_flydsl_communicator_stays_on_hip(self):
        quick_reduce = _make_quick_reduce()
        hip_calls = []
        with patch.object(
            qr_mod.ops,
            "qr_all_reduce",
            lambda *args: hip_calls.append(args),
            create=True,
        ):
            quick_reduce.quick_all_reduce(_bf16(MB))
        self.assertEqual(len(hip_calls), 1)


class TestFlyDSLInt4Window(CustomTestCase):
    """``uses_flydsl_int4`` is the dispatcher's question, so it owns the size
    window: the custom all-reduce wins below the floor, FlyDSL above it."""

    def _quick_reduce(self, world_size=8, min_size=2 * MB):
        quick_reduce = _make_quick_reduce(world_size=world_size)
        quick_reduce._flydsl_int4 = _FakeFlyDSLInt4()
        quick_reduce._flydsl_min_size = min_size
        return quick_reduce

    def test_window_starts_at_the_floor(self):
        quick_reduce = self._quick_reduce(min_size=2 * MB)
        below = _bf16(MB - 8)  # 2 MiB less 16 B
        at = _bf16(MB)
        self.assertFalse(quick_reduce.uses_flydsl_int4(below))
        self.assertTrue(quick_reduce.uses_flydsl_int4(at))
        self.assertTrue(quick_reduce.uses_flydsl_int4(_bf16(64 * MB)))

    def test_window_is_wider_than_the_hip_window_with_bf16_table(self):
        # should_quick_allreduce reads the BF16 table when the fp16 cast is
        # off, where TP8 INT4 needs 2048 MB: HIP would take nothing. FlyDSL
        # ignores the cast, so its window must not follow that table.
        quick_reduce = self._quick_reduce()
        quick_reduce.use_fp16_kernels = 0
        inp = _bf16(4 * MB)  # 8 MiB
        self.assertFalse(quick_reduce.should_quick_allreduce(inp))
        self.assertTrue(quick_reduce.uses_flydsl_int4(inp))

    def test_max_size_caps_the_window(self):
        quick_reduce = self._quick_reduce()
        quick_reduce.qr_max_size = 16 * MB
        self.assertTrue(quick_reduce.uses_flydsl_int4(_bf16(8 * MB)))  # 16 MiB
        self.assertFalse(quick_reduce.uses_flydsl_int4(_bf16(8 * MB + 8)))

    def test_four_gib_window_of_the_kernel_caps_the_window(self):
        quick_reduce = self._quick_reduce()
        quick_reduce.qr_max_size = 8 * 1024 * MB
        quick_reduce._FLYDSL_MAX_SIZE = 8 * MB  # stand-in for 4 GiB
        self.assertTrue(quick_reduce.uses_flydsl_int4(_bf16(4 * MB)))
        self.assertFalse(quick_reduce.uses_flydsl_int4(_bf16(4 * MB + 8)))

    def test_payloads_flydsl_cannot_take_are_not_claimed(self):
        quick_reduce = self._quick_reduce()
        for name, inp in (
            ("fp16", torch.empty(MB, dtype=torch.float16)),
            ("non_contiguous", _bf16(2 * MB).view(1024, 2048).t()),
            ("misaligned", _bf16(MB + 16)[4 : MB + 4]),
            ("not_16_bytes", _bf16(MB + 3)),
        ):
            with self.subTest(name):
                self.assertFalse(quick_reduce.uses_flydsl_int4(inp))

    def test_disabled_or_missing_communicator_claims_nothing(self):
        quick_reduce = self._quick_reduce()
        quick_reduce.disabled = True
        self.assertFalse(quick_reduce.uses_flydsl_int4(_bf16(MB)))
        quick_reduce = _make_quick_reduce()  # no FlyDSL communicator
        self.assertFalse(quick_reduce.uses_flydsl_int4(_bf16(MB)))

    def test_claimed_payload_runs_on_flydsl(self):
        # The dispatcher resolves to "qr" on uses_flydsl_int4; quick_all_reduce
        # must then really take the FlyDSL branch, not HIP.
        quick_reduce = self._quick_reduce()
        inp = _bf16(MB)
        self.assertTrue(quick_reduce.uses_flydsl_int4(inp))
        hip_calls = []
        with patch.object(
            qr_mod.ops,
            "qr_all_reduce",
            lambda *args: hip_calls.append(args),
            create=True,
        ):
            quick_reduce.quick_all_reduce(inp)
        self.assertEqual(quick_reduce._flydsl_int4.allreduce_calls, 1)
        self.assertEqual(hip_calls, [])


class TestFlyDSLInt4Init(CustomTestCase):
    def setUp(self):
        _FakeFlyDSLInt4.instances.clear()
        fake_flydsl = types.ModuleType("aiter.ops.flydsl")
        fake_flydsl.QuickAllReduceInt4 = _FakeFlyDSLInt4
        self._modules = patch.dict(sys.modules, {"aiter.ops.flydsl": fake_flydsl})
        self._modules.start()
        self._arch = patch.object(qr_mod, "flydsl_int4_arch_available", lambda: True)
        self._arch.start()
        self._sync = patch.object(torch.cuda, "synchronize", lambda *args: None)
        self._sync.start()

    def tearDown(self):
        self._sync.stop()
        self._arch.stop()
        self._modules.stop()

    def _init(
        self,
        quant_level=QuickReduceRegime.INT4,
        use_aiter=True,
        flydsl=True,
        world_size=8,
        min_size_kb=0,
        use_fp16_kernels=1,
    ):
        quick_reduce = _make_quick_reduce(quant_level, world_size=world_size)
        quick_reduce.use_fp16_kernels = use_fp16_kernels
        with (
            envs.SGLANG_USE_AITER.override(use_aiter),
            envs.SGLANG_ROCM_FLYDSL_QUICK_REDUCE.override(flydsl),
            envs.SGLANG_ROCM_FLYDSL_QUICK_REDUCE_MIN_SIZE_KB.override(min_size_kb),
        ):
            quick_reduce._init_flydsl_int4()
        return quick_reduce

    def test_int4_with_aiter_builds_and_compiles(self):
        quick_reduce = self._init()
        fake = quick_reduce._flydsl_int4
        self.assertIsInstance(fake, _FakeFlyDSLInt4)
        self.assertEqual(fake.init_kwargs["rank"], 0)
        self.assertEqual(fake.init_kwargs["world_size"], 8)
        self.assertIs(fake.init_kwargs["group"], quick_reduce.group)
        self.assertEqual(fake.compiled, [(torch.bfloat16, MB)])

    def test_min_size_defaults_per_world_size(self):
        # The FP16 INT4 row of _QR_MIN_SIZE: 1 MiB at TP2, 2 MiB at TP4/TP8.
        for world_size, expected in ((2, MB), (4, 2 * MB), (8, 2 * MB)):
            with self.subTest(world_size=world_size):
                quick_reduce = self._init(world_size=world_size)
                self.assertEqual(quick_reduce._flydsl_min_size, expected)

    def test_min_size_ignores_the_bf16_cast_setting(self):
        # With the cast off the BF16 table asks 2048 MB at TP8; FlyDSL does
        # not cast, so its floor must stay the FP16 row.
        quick_reduce = self._init(use_fp16_kernels=0)
        self.assertEqual(quick_reduce._flydsl_min_size, 2 * MB)

    def test_min_size_env_overrides_the_default(self):
        quick_reduce = self._init(min_size_kb=8192)
        self.assertEqual(quick_reduce._flydsl_min_size, 8 * MB)

    def test_other_codecs_do_not_build(self):
        for level in (QuickReduceRegime.FP, QuickReduceRegime.INT6):
            with self.subTest(level=level):
                self.assertIsNone(self._init(quant_level=level)._flydsl_int4)
        self.assertEqual(_FakeFlyDSLInt4.instances, [])

    def test_aiter_off_does_not_build(self):
        self.assertIsNone(self._init(use_aiter=False)._flydsl_int4)

    def test_opt_out_does_not_build(self):
        self.assertIsNone(self._init(flydsl=False)._flydsl_int4)

    def test_unsupported_arch_does_not_build(self):
        with patch.object(qr_mod, "flydsl_int4_arch_available", lambda: False):
            self.assertIsNone(self._init()._flydsl_int4)

    def test_missing_aiter_kernel_falls_back_to_hip(self):
        # An AITER without QuickAllReduceInt4 (older than ROCm/aiter#4970).
        with patch.dict(
            sys.modules, {"aiter.ops.flydsl": types.ModuleType("aiter.ops.flydsl")}
        ):
            self.assertIsNone(self._init()._flydsl_int4)

    def test_close_closes_flydsl(self):
        quick_reduce = _make_quick_reduce()
        quick_reduce.disabled = True  # skip the HIP teardown
        fake = _FakeFlyDSLInt4()
        quick_reduce._flydsl_int4 = fake

        quick_reduce.close()

        self.assertTrue(fake.closed)
        self.assertIsNone(quick_reduce._flydsl_int4)


if __name__ == "__main__":
    unittest.main()
