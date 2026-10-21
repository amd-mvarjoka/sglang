"""SGLang QuickAllReduce routed to AITER's FlyDSL INT4 all-reduce.

AITER's own test covers the INT4 codec. This one covers what SGLang adds on
top: the routing in QuickAllReduce, replaying the kernel from a captured CUDA
graph, and ranks agreeing bit for bit on the reduced output.
"""

import importlib.util
import multiprocessing
import os
import socket
import time
import unittest

import torch
import torch.distributed as dist

from sglang.srt.distributed.device_communicators.quick_all_reduce import (
    QuickAllReduce,
    flydsl_int4_arch_available,
)
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-c-test-4-gpu-amd")
register_amd_ci(est_time=60, suite="stage-c-test-large-8-gpu-amd-mi35x")

# Same bounds as AITER's QuickAllReduceInt4 test: the INT4 group-16 codec
# against an fp32 all-reduce. A 32 KiB tile the kernel never wrote scores ~0 dB.
SQNR_MIN_DB = 18.0
TILE_SQNR_MIN_DB = 8.0
TILE_ELEMS = 32 * 1024 // 2

# (tokens, hidden). (3, 7168) is not a whole number of tiles; (9216, 4096)
# needs more tiles than the persistent grid, so it runs the super-tile kernel.
SHAPES = [(3, 7168), (512, 7168), (9216, 4096)]


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _flydsl_int4_importable() -> bool:
    if importlib.util.find_spec("aiter") is None:
        return False
    try:
        from aiter.ops.flydsl import QuickAllReduceInt4  # noqa: F401
    except ImportError:
        return False
    return True


def _rank_input(shape, rank: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed * 1000 + rank)
    return (torch.randn(shape, generator=gen) * 0.1).to(torch.bfloat16)


def _reference(shape, world_size: int, seed: int) -> torch.Tensor:
    ref = torch.zeros(shape, dtype=torch.float32)
    for rank in range(world_size):
        ref += _rank_input(shape, rank, seed).float()
    return ref


def _sqnr_db(got: torch.Tensor, ref: torch.Tensor) -> float:
    mse = ((got - ref) ** 2).mean()
    return float(10.0 * torch.log10((ref * ref).mean() / mse))


def _min_tile_sqnr_db(got: torch.Tensor, ref: torch.Tensor) -> float:
    g, r = got.reshape(-1), ref.reshape(-1)
    n_full = (g.numel() // TILE_ELEMS) * TILE_ELEMS
    vals = []
    if n_full:
        gt, rt = g[:n_full].view(-1, TILE_ELEMS), r[:n_full].view(-1, TILE_ELEMS)
        mse = ((gt - rt) ** 2).mean(dim=1)
        vals.append(float((10.0 * torch.log10((rt * rt).mean(dim=1) / mse)).min()))
    if g.numel() > n_full:
        vals.append(_sqnr_db(g[n_full:], r[n_full:]))
    return min(vals)


def _check(
    out: torch.Tensor,
    shape,
    world_size: int,
    seed: int,
    what: str,
    sqnr_min_db: float = SQNR_MIN_DB,
) -> None:
    got = out.float().cpu()
    ref = _reference(shape, world_size, seed)
    assert torch.isfinite(got).all(), f"{what}: non-finite output"
    sqnr = _sqnr_db(got, ref)
    assert sqnr >= sqnr_min_db, f"{what}: SQNR {sqnr:.2f} dB < {sqnr_min_db}"
    tile_sqnr = _min_tile_sqnr_db(got, ref)
    assert tile_sqnr >= TILE_SQNR_MIN_DB, (
        f"{what}: min-tile SQNR {tile_sqnr:.2f} dB < {TILE_SQNR_MIN_DB}"
    )
    # Every rank dequantizes the same all-gathered bytes.
    checksum = torch.tensor(
        [int(out.view(torch.int16).long().sum().item())], dtype=torch.int64
    )
    checksums = [torch.zeros_like(checksum) for _ in range(world_size)]
    dist.all_gather(checksums, checksum)
    assert all(torch.equal(c, checksums[0]) for c in checksums), (
        f"{what}: ranks disagree on the reduced output"
    )


def _run_flydsl_int4_test(rank: int, world_size: int, port: int) -> None:
    os.environ["ROCM_QUICK_REDUCE_QUANTIZATION"] = "INT4"
    os.environ["ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16"] = "1"
    os.environ["SGLANG_USE_AITER"] = "1"
    os.environ.pop("SGLANG_ROCM_FLYDSL_QUICK_REDUCE", None)

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        qr = QuickAllReduce(group=dist.group.WORLD, device=device)
        assert not qr.disabled
        assert qr._flydsl_int4 is not None, "FlyDSL INT4 communicator not built"
        try:
            for seed, shape in enumerate(SHAPES):
                inp = _rank_input(shape, rank, seed).to(device)
                # The dispatcher's window: under the floor the custom
                # all-reduce keeps the payload; quick_all_reduce itself still
                # runs FlyDSL when called directly, which is what is checked.
                nbytes = inp.numel() * inp.element_size()
                assert qr.uses_flydsl_int4(inp) == (nbytes >= qr._flydsl_min_size), (
                    f"window wrong for {shape}: {nbytes} B, floor "
                    f"{qr._flydsl_min_size} B"
                )
                dist.barrier()
                out = qr.quick_all_reduce(inp)
                torch.cuda.synchronize()
                _check(out, shape, world_size, seed, f"eager {shape}")

            # Capture once, then replay on new data written into the same
            # input buffer, as SGLang's CUDA graph runner does.
            shape = (512, 7168)
            inp = torch.empty(shape, dtype=torch.bfloat16, device=device)
            out = torch.empty_like(inp)
            graph = torch.cuda.CUDAGraph()
            dist.barrier()
            with torch.cuda.graph(graph):
                qr.quick_all_reduce(inp, out=out)
            for seed in (100, 101, 102):
                inp.copy_(_rank_input(shape, rank, seed))
                torch.cuda.synchronize()
                dist.barrier()
                graph.replay()
                torch.cuda.synchronize()
                _check(out, shape, world_size, seed, f"graph replay {seed=}")

            # The kernel reduces in fp16. BF16 inputs or sums beyond fp16 range
            # are lossy but must stay finite (cf. the HIP fix in
            # test_quick_allreduce_bf16_range.py).
            for value, mixed in ((20_000.0, False), (80_000.0, False), (1.0, True)):
                big = torch.full((1 << 20,), value, dtype=torch.bfloat16, device=device)
                if mixed:
                    big[::32] = 80_000.0
                dist.barrier()
                out = qr.quick_all_reduce(big)
                torch.cuda.synchronize()
                assert torch.isfinite(out).all().item(), (
                    f"{value=} {mixed=}: non-finite output"
                )

            # FP16 payloads stay on the HIP INT4 kernels. Its codec differs,
            # so the bound only catches a broken route, not codec quality.
            fp16 = _rank_input(shape, rank, 200).to(device, torch.float16)
            assert not qr._flydsl_accepts(fp16, torch.empty_like(fp16))
            assert not qr.uses_flydsl_int4(fp16)
            dist.barrier()
            out = qr.quick_all_reduce(fp16)
            torch.cuda.synchronize()
            _check(out, shape, world_size, 200, "fp16 via HIP", sqnr_min_db=12.0)
        finally:
            qr.close()
    finally:
        dist.destroy_process_group()


class TestQuickAllReduceFlyDSLInt4(CustomTestCase):
    @unittest.skipUnless(
        flydsl_int4_arch_available()
        and torch.cuda.device_count() >= 4
        and _flydsl_int4_importable(),
        "FlyDSL INT4 QuickReduce test requires four gfx942/gfx950 GPUs and "
        "an AITER with QuickAllReduceInt4",
    )
    def test_flydsl_int4(self):
        world_size = 8 if torch.cuda.device_count() >= 8 else 4
        port = _get_open_port()
        context = multiprocessing.get_context("spawn")
        processes = [
            context.Process(
                target=_run_flydsl_int4_test,
                args=(rank, world_size, port),
            )
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()

        # The first FlyDSL launch JIT-compiles the kernels.
        deadline = time.monotonic() + 300
        for process in processes:
            process.join(max(0, deadline - time.monotonic()))

        if any(process.is_alive() for process in processes):
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join()
            self.fail("FlyDSL INT4 QuickReduce test timed out")

        for rank, process in enumerate(processes):
            self.assertEqual(
                process.exitcode,
                0,
                f"FlyDSL INT4 QuickReduce test failed on rank {rank}",
            )


if __name__ == "__main__":
    unittest.main()
