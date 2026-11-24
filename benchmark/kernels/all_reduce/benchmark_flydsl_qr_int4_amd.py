"""
Sweep AITER's FlyDSL INT4 QuickReduce against SGLang's default all-reduce path.

QuickAllReduce hands a payload to FlyDSL from a size floor up
(SGLANG_ROCM_FLYDSL_QUICK_REDUCE_MIN_SIZE_KB; the default is the FP16 INT4 row
of QuickAllReduce._QR_MIN_SIZE, 1 MiB at TP2 and 2 MiB at TP4/TP8). That floor
comes from kernel timings on other shapes. This script measures where it
belongs for the shape you serve, on the all-reduce SGLang really runs.

Two call sites are timed for every token count, each in eager mode (prefill)
and CUDA-graph mode (decode):

- ``ar``:    tensor_model_parallel_all_reduce(x)
- ``fused``: what a layer does with AITER all-reduce fusion on:
             tensor_model_parallel_fused_allreduce_rmsnorm when it returns a
             result, else all-reduce followed by AITER's add + RMSNorm. FlyDSL
             has no fused kernel, so on its arm this is always the second form.

Both arms run in one process, on the same TP group, by moving the floor that
QuickAllReduce.uses_flydsl_int4 compares against:

- ``default``: floor = infinity, FlyDSL never claims a payload. This is the
               path without FlyDSL (custom all-reduce, fused kernel, and HIP
               INT4 quick-reduce for payloads above the custom-AR limit).
- ``flydsl``:  floor = 1 byte, FlyDSL claims every payload it can take.

It also reports FlyDSL's SQNR against an fp32 all-reduce (INT4 is lossy; aiter's
own test requires >= 18 dB), and from the speedups derives the smallest payload
from which FlyDSL stays ahead at every larger size, per call site and mode.

Eager timings of small payloads are launch-bound; read decode-sized rows from
graph mode and prefill-sized rows from eager mode.

Run it with the INT4 codec and AITER on, one process per GPU:

  ROCM_QUICK_REDUCE_QUANTIZATION=INT4 SGLANG_USE_AITER=1 \\
  torchrun --nproc_per_node=8 \\
    benchmark/kernels/all_reduce/benchmark_flydsl_qr_int4_amd.py \\
    --hidden 7168 --csv-out /tmp/flydsl_qr_int4_tp8.csv

Run it once per TP size you serve; the floor depends on it.
"""

import argparse
import csv
import os
import statistics
import sys
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from sglang.srt.distributed.communication_op import (
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_fused_allreduce_rmsnorm,
)
from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_tp_group,
    graph_capture,
    init_distributed_environment,
    initialize_model_parallel,
    set_custom_all_reduce,
)
from sglang.test.test_utils import publish_build_topology

DEFAULT_TOKENS = (
    "1,2,4,8,16,32,64,128,192,256,384,512,768,1024,1536,2048,3072,4096,"
    "6144,8192,12288,16384,32768"
)
NEVER = sys.maxsize  # a floor no payload reaches: FlyDSL claims nothing
ALWAYS = 1  # a floor every payload reaches: FlyDSL claims all it can take
SITES = ("ar", "fused")
MODES = ("eager", "graph")


def _barrier(device: torch.device):
    try:
        dist.barrier(device_ids=[device.index])
    except TypeError:
        dist.barrier()


def _mean_across_ranks(value: float, device: torch.device) -> float:
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return float(t.item())


def _min_across_ranks(value: float, device: torch.device) -> float:
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return float(t.item())


def _time_us(
    fn: Callable[[], None], warmup: int, iters: int, repeats: int, device
) -> float:
    """Median over repeats of the mean per-call time of ``iters`` calls."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    samples: List[float] = []
    for _ in range(max(1, repeats)):
        _barrier(device)
        torch.cuda.synchronize()
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / iters)
    return float(statistics.median(samples))


def _time_graph_us(
    fn: Callable[[], None],
    graph_ops: int,
    warmup: int,
    iters: int,
    repeats: int,
    device,
) -> float:
    """Per-call time of ``fn`` replayed ``graph_ops`` times from one CUDA graph.

    Several calls per graph keep the graph-launch cost out of a kernel that
    takes tens of microseconds.
    """
    fn()  # outside the capture: first-use registration and allocator warmup
    torch.cuda.synchronize()
    with graph_capture() as gc:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=gc.stream):
            for _ in range(graph_ops):
                fn()
    return _time_us(graph.replay, warmup, iters, repeats, device) / graph_ops


def _sqnr_db(got: torch.Tensor, ref: torch.Tensor) -> float:
    mse = ((got.float() - ref) ** 2).mean()
    return float(10.0 * torch.log10((ref * ref).mean() / mse))


def _make_call_sites(
    tokens: int, hidden: int, eps: float, rank: int, device: torch.device
) -> Tuple[Dict[str, Callable[[], None]], torch.Tensor]:
    from aiter import rmsnorm2d_fwd_with_add as fused_add_rms_norm

    torch.manual_seed(1234 + rank * 17)
    x = torch.randn((tokens, hidden), dtype=torch.float32, device=device).bfloat16()
    residual = torch.randn_like(x)
    weight = torch.randn((hidden,), dtype=torch.float32, device=device).bfloat16()
    out = torch.empty_like(x)
    residual_out = torch.empty_like(x)

    def ar():
        tensor_model_parallel_all_reduce(x)

    def fused():
        if (
            tensor_model_parallel_fused_allreduce_rmsnorm(x, residual, weight, eps)
            is None
        ):
            # What the layer does when the fused kernel declines.
            reduced = tensor_model_parallel_all_reduce(x)
            fused_add_rms_norm(out, reduced, residual, residual_out, weight, eps)

    return {"ar": ar, "fused": fused}, x


def _smallest_winning_bytes(
    rows: List[Dict], site: str, mode: str, min_speedup: float
) -> Optional[int]:
    """Smallest payload from which FlyDSL wins at every larger size tested."""
    pts = sorted(
        ((r["bytes_per_rank"], r[f"{site}_speedup"]) for r in rows if r["mode"] == mode)
    )
    floor = None
    for nbytes, speedup in reversed(pts):
        if speedup < min_speedup:
            break
        floor = nbytes
    return floor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep FlyDSL INT4 QuickReduce against the default all-reduce."
    )
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--tokens", type=str, default=DEFAULT_TOKENS)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--graph-ops",
        type=int,
        default=10,
        help="Calls captured per CUDA graph in graph mode.",
    )
    parser.add_argument(
        "--mode", type=str, default="both", choices=("eager", "graph", "both")
    )
    parser.add_argument(
        "--min-speedup",
        type=float,
        default=1.03,
        help="A size counts as a FlyDSL win only above this speedup (noise margin).",
    )
    parser.add_argument("--csv-out", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    torch.cuda.set_device(local_rank % torch.cuda.device_count())
    device = torch.device(f"cuda:{local_rank % torch.cuda.device_count()}")

    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
        backend="nccl",
    )
    publish_build_topology(world_rank=rank, tp_size=world_size)
    initialize_model_parallel()

    tp_group = get_tp_group()
    qr_comm = tp_group.qr_comm
    flydsl_ready = (
        qr_comm is not None
        and not qr_comm.disabled
        and getattr(qr_comm, "_flydsl_int4", None) is not None
    )
    # Every rank must agree before any collective below, or ranks hang.
    if _min_across_ranks(float(flydsl_ready), device) < 1.0:
        if rank == 0:
            print(
                "FlyDSL INT4 QuickReduce is not active. Needs "
                "ROCM_QUICK_REDUCE_QUANTIZATION=INT4, SGLANG_USE_AITER=1, "
                "SGLANG_ROCM_FLYDSL_QUICK_REDUCE unset or 1, gfx942/gfx950, "
                "TP 2/4/8, and an AITER with QuickAllReduceInt4."
            )
        destroy_model_parallel()
        destroy_distributed_environment()
        sys.exit(1)

    default_floor = qr_comm._flydsl_min_size
    token_list = [int(t) for t in args.tokens.split(",") if t.strip()]
    modes = MODES if args.mode == "both" else (args.mode,)
    rows: List[Dict] = []

    if rank == 0:
        print(
            f"Config: tp={world_size} hidden={args.hidden} dtype=bf16 "
            f"default floor={default_floor // 1024} KiB warmup={args.warmup} "
            f"iters={args.iters} repeats={args.repeats} graph_ops={args.graph_ops}"
        )

    for mode in modes:
        if rank == 0:
            print(f"\n=== {mode} ===")
            print(
                "| tokens | KiB/rank | ar default | ar flydsl | ar x | "
                "fused default | fused flydsl | fused x | SQNR dB |"
            )
            print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for tokens in token_list:
            sites, x = _make_call_sites(tokens, args.hidden, args.eps, rank, device)
            nbytes = x.numel() * x.element_size()
            timings: Dict[str, float] = {}
            for arm, floor in (("default", NEVER), ("flydsl", ALWAYS)):
                qr_comm._flydsl_min_size = floor
                # An arm that does not run what it names would report a
                # meaningless speedup: fail loudly instead.
                assert qr_comm.uses_flydsl_int4(x) == (arm == "flydsl"), (
                    f"{arm} arm: FlyDSL routing is "
                    f"{qr_comm.uses_flydsl_int4(x)} for {tokens}x{args.hidden}"
                )
                for site in SITES:
                    fn = sites[site]
                    if mode == "eager":
                        us = _time_us(fn, args.warmup, args.iters, args.repeats, device)
                    else:
                        us = _time_graph_us(
                            fn,
                            args.graph_ops,
                            args.warmup,
                            args.iters,
                            args.repeats,
                            device,
                        )
                    timings[f"{site}_{arm}"] = _mean_across_ranks(us, device)

            # Accuracy of the FlyDSL arm against an exact fp32 all-reduce.
            ref = x.float()
            dist.all_reduce(ref, group=tp_group.device_group)
            qr_comm._flydsl_min_size = ALWAYS
            sqnr = _sqnr_db(tensor_model_parallel_all_reduce(x.clone()), ref)
            qr_comm._flydsl_min_size = default_floor

            row = {
                "mode": mode,
                "tokens": tokens,
                "hidden": args.hidden,
                "bytes_per_rank": nbytes,
                "sqnr_db": sqnr,
                **timings,
                "ar_speedup": timings["ar_default"] / timings["ar_flydsl"],
                "fused_speedup": timings["fused_default"] / timings["fused_flydsl"],
            }
            rows.append(row)
            if rank == 0:
                print(
                    f"| {tokens} | {nbytes // 1024} | {row['ar_default']:.1f} | "
                    f"{row['ar_flydsl']:.1f} | {row['ar_speedup']:.2f} | "
                    f"{row['fused_default']:.1f} | {row['fused_flydsl']:.1f} | "
                    f"{row['fused_speedup']:.2f} | {sqnr:.1f} |"
                )
            del sites, x, ref

    if rank == 0:
        print(
            f"\nSmallest payload from which FlyDSL wins by >= {args.min_speedup}x "
            "at every larger size tested (default floor "
            f"{default_floor // 1024} KiB):"
        )
        for mode in modes:
            for site in SITES:
                floor = _smallest_winning_bytes(rows, site, mode, args.min_speedup)
                if floor is None:
                    print(
                        f"  {mode:5s} {site:5s}: FlyDSL is not ahead at the largest size"
                    )
                else:
                    kib = (floor + 1023) // 1024
                    print(
                        f"  {mode:5s} {site:5s}: {kib} KiB  "
                        f"(SGLANG_ROCM_FLYDSL_QUICK_REDUCE_MIN_SIZE_KB={kib})"
                    )
        print(
            "\nThe floor serves both call sites: take the larger of the two, "
            "and the 'fused' one if AITER all-reduce fusion is on. SQNR below "
            "18 dB would fail aiter's own accuracy bound."
        )
        if args.csv_out:
            os.makedirs(os.path.dirname(args.csv_out) or ".", exist_ok=True)
            with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"Saved CSV to: {args.csv_out}")

    _barrier(device)
    destroy_model_parallel()
    destroy_distributed_environment()


if __name__ == "__main__":
    main()
