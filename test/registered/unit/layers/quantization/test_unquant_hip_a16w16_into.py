"""a16w16_into picks the library AITER's bf16 table tuned for the shape, and keeps
the FlyDSL tile for an fp32 output, which the table keys separately."""

import sys
import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization import unquant_hip
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

N, K = 64, 32
_FLYDSL_PARAMS = dict.fromkeys(
    (
        "block_m",
        "block_n",
        "block_k",
        "stages",
        "split_k",
        "m_waves",
        "n_waves",
        "k_waves",
        "group_m",
        "use_half_tile_interleaved",
    ),
    1,
)


class TestA16W16Into(CustomTestCase):
    def setUp(self):
        self.ran = []
        self.table = {}

        def tuned_config(m, n, k, bias, dtype, otype):
            if (n, k, otype) != (N, K, "torch.bfloat16") or m not in self.table:
                return None
            return {"libtype": self.table[m], "kernelName": "k"}

        def write(name, x, w, out):
            self.ran.append(name)
            out.copy_(x.float() @ w.float().t())

        def tgemm_mm(x, w, bias=None, otype=None, out=None):
            write("tgemm", x, w, out)
            return out

        def flydsl_gemm(x, w_t, out, user_kwargs, stream, layout, out_dtype):
            write("flydsl", x, w_t.t(), out)

        tuned_gemm = types.ModuleType("aiter.tuned_gemm")
        tuned_gemm.get_GEMM_A16W16_tuned_config = tuned_config
        tuned_gemm.tgemm = types.SimpleNamespace(mm=tgemm_mm)
        flydsl = types.ModuleType("aiter.ops.flydsl")
        flydsl.gemm_kernels = types.SimpleNamespace(
            get_flydsl_hgemm_kernel_params=lambda name: _FLYDSL_PARAMS
        )
        gfx950 = types.ModuleType("aiter.ops.flydsl.kernels.gemm_a16w16_gfx950")
        gfx950.gemm_a16w16 = flydsl_gemm
        modules = {
            "aiter": types.ModuleType("aiter"),
            "aiter.tuned_gemm": tuned_gemm,
            "aiter.ops": types.ModuleType("aiter.ops"),
            "aiter.ops.flydsl": flydsl,
            "aiter.ops.flydsl.kernels": types.ModuleType("aiter.ops.flydsl.kernels"),
            "aiter.ops.flydsl.kernels.gemm_a16w16_gfx950": gfx950,
        }
        for p in (
            patch.dict(sys.modules, modules),
            patch.object(unquant_hip, "_use_aiter", True),
            patch.object(unquant_hip, "_is_gfx95", True),
            patch("torch.cuda.current_stream", lambda device=None: None),
        ):
            p.start()
            self.addCleanup(p.stop)
        for cached in (
            unquant_hip._a16w16_tuned_libtype,
            unquant_hip._a16w16_flydsl_launch,
        ):
            cached.cache_clear()
            self.addCleanup(cached.cache_clear)

    def _run(self, m, out_dtype):
        x = torch.randn(m, K, dtype=torch.bfloat16)
        w = torch.randn(N, K, dtype=torch.bfloat16)
        out = torch.full((m, N), float("nan"), dtype=out_dtype)
        if unquant_hip.a16w16_into(x, w, out):
            torch.testing.assert_close(
                out.float(), x.float() @ w.float().t(), rtol=2e-2, atol=2e-1
            )
            return self.ran
        return None

    def test_routing(self):
        # (rows tuned in the table, M, out dtype) -> library that wrote out, None: torch.mm
        cases = [
            ({16: "triton"}, 16, torch.bfloat16, ["tgemm"]),
            ({16: "asm"}, 16, torch.bfloat16, ["tgemm"]),
            ({16: "torch"}, 16, torch.bfloat16, ["tgemm"]),
            ({16: "flydsl"}, 16, torch.bfloat16, ["flydsl"]),
            ({16: "triton", 8: "flydsl"}, 16, torch.float32, ["flydsl"]),
            ({8: "flydsl"}, 16, torch.bfloat16, ["flydsl"]),
            ({16: "triton"}, 16, torch.float32, None),
            ({}, 16, torch.bfloat16, None),
        ]
        for table, m, out_dtype, expected in cases:
            with self.subTest(table=table, m=m, out_dtype=out_dtype):
                self.ran.clear()
                self.table = table
                unquant_hip._a16w16_tuned_libtype.cache_clear()
                unquant_hip._a16w16_flydsl_launch.cache_clear()
                self.assertEqual(self._run(m, out_dtype), expected)

    def test_hipblaslt_window_stays_on_torch(self):
        self.table = {256: "triton"}
        with patch.object(unquant_hip, "_HIPBLASLT_SHAPES", frozenset({(N, K)})):
            self.assertIsNone(self._run(256, torch.bfloat16))


if __name__ == "__main__":
    unittest.main()
