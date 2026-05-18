#!/usr/bin/env python3
"""Benchmark cache-hit behavior as concurrent users increase.

Concurrency here means a wave of requests submitted together to nano-vLLM.
Every request in a wave shares the same full-block prefix, so the reported
hit_ratio should show whether the backend is benefiting from prefix cache.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass, field
from statistics import median
from time import perf_counter
from typing import Any

from benchmark_kvcache_backends import (
    DEFAULT_BACKENDS,
    RESULT_PREFIX,
    csv_list,
    cuda_sync,
    default_model_path,
    ensure_model,
    fmt,
    get_vocab_size,
    make_shared_prefix,
    maybe_cuda_peak_memory,
    token_id,
)


@dataclass
class WaveMetrics:
    concurrency: int
    rounds: int
    wall_time_s: float = 0.0
    prefill_time_s: float = 0.0
    decode_time_s: float = 0.0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    compute_prompt_tokens: int = 0
    generated_tokens: int = 0
    request_latencies_s: list[float] = field(default_factory=list)

    @property
    def hit_ratio(self) -> float:
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0

    @property
    def request_per_s(self) -> float:
        total = self.concurrency * self.rounds
        return total / self.wall_time_s if self.wall_time_s > 0 else 0.0

    @property
    def output_tok_s(self) -> float:
        return self.generated_tokens / self.wall_time_s if self.wall_time_s > 0 else 0.0

    @property
    def decode_tok_s(self) -> float:
        return (
            self.generated_tokens / self.decode_time_s
            if self.decode_time_s > 0
            else 0.0
        )

    @property
    def p50_latency_ms(self) -> float:
        if not self.request_latencies_s:
            return 0.0
        return median(self.request_latencies_s) * 1000.0

    @property
    def p95_latency_ms(self) -> float:
        if not self.request_latencies_s:
            return 0.0
        ordered = sorted(self.request_latencies_s)
        idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
        return ordered[idx] * 1000.0

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["hit_ratio"] = self.hit_ratio
        data["request_per_s"] = self.request_per_s
        data["output_tok_s"] = self.output_tok_s
        data["decode_tok_s"] = self.decode_tok_s
        data["p50_latency_ms"] = self.p50_latency_ms
        data["p95_latency_ms"] = self.p95_latency_ms
        return data


@dataclass
class BackendResult:
    ok: bool
    backend: str
    waves: dict[str, dict[str, Any]] = field(default_factory=dict)
    peak_memory_allocated_gb: float | None = None
    peak_memory_reserved_gb: float | None = None
    error: str | None = None
    traceback: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default=default_model_path())
    parser.add_argument("--hf-model-id", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--download-if-missing", action="store_true")
    parser.add_argument("--backends", type=str, default=DEFAULT_BACKENDS)
    parser.add_argument("--concurrency-levels", type=str, default="1,2,4,8,16,32")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--prefix-len", type=int, default=512)
    parser.add_argument("--suffix-len", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--eager",
        action="store_true",
        help="Disable CUDA graph decode and run every decode step eagerly.",
    )
    parser.add_argument(
        "--cuda-graphs",
        dest="eager",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--json-out", type=str, default=None)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--backend", type=str, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def int_csv(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    return [value for value in values if value > 0]


def make_wave_prompts(
    concurrency: int,
    round_idx: int,
    shared_prefix: list[int],
    suffix_len: int,
    vocab_size: int,
) -> list[list[int]]:
    prompts: list[list[int]] = []
    for user_idx in range(concurrency):
        salt = round_idx * 100_000 + user_idx * 997
        suffix = [token_id(30_000 + salt, i, vocab_size) for i in range(suffix_len)]
        prompts.append(shared_prefix + suffix)
    return prompts


def run_wave(llm: Any, prompts: list[list[int]], sampling_params: Any) -> WaveMetrics:
    t0 = perf_counter()
    start_times: dict[int, float] = {}
    for prompt in prompts:
        llm.add_request(prompt, sampling_params)
        seq = llm.scheduler.waiting[-1]
        start_times[seq.seq_id] = t0

    metrics = WaveMetrics(concurrency=len(prompts), rounds=1)
    while not llm.is_finished():
        seqs, is_prefill = llm.scheduler.schedule()
        if is_prefill:
            prompt_tokens = sum(len(seq) for seq in seqs)
            cached_tokens = sum(seq.num_cached_tokens for seq in seqs)
            compute_tokens = sum(len(seq) - seq.num_cached_tokens for seq in seqs)
        else:
            prompt_tokens = cached_tokens = compute_tokens = 0

        cuda_sync()
        step_start = perf_counter()
        token_ids = llm.model_runner.call("run", seqs, is_prefill)
        cuda_sync()
        step_time = perf_counter() - step_start

        llm.scheduler.postprocess(seqs, token_ids)
        now = perf_counter()
        for seq in seqs:
            if seq.is_finished and seq.seq_id in start_times:
                metrics.request_latencies_s.append(now - start_times.pop(seq.seq_id))

        if is_prefill:
            metrics.prompt_tokens += prompt_tokens
            metrics.cached_tokens += cached_tokens
            metrics.compute_prompt_tokens += compute_tokens
            metrics.prefill_time_s += step_time
        else:
            metrics.generated_tokens += len(seqs)
            metrics.decode_time_s += step_time

    metrics.wall_time_s = perf_counter() - t0
    return metrics


def merge_wave(total: WaveMetrics, wave: WaveMetrics) -> None:
    total.wall_time_s += wave.wall_time_s
    total.prefill_time_s += wave.prefill_time_s
    total.decode_time_s += wave.decode_time_s
    total.prompt_tokens += wave.prompt_tokens
    total.cached_tokens += wave.cached_tokens
    total.compute_prompt_tokens += wave.compute_prompt_tokens
    total.generated_tokens += wave.generated_tokens
    total.request_latencies_s.extend(wave.request_latencies_s)


def run_worker(args: argparse.Namespace) -> BackendResult:
    try:
        import torch
        from nanovllm import LLM, SamplingParams

        os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
        os.environ.setdefault("NANOVLLM_SAW_INT4_HADAMARD_ORDER", "16")

        levels = int_csv(args.concurrency_levels)
        if not args.eager and levels:
            graph_limit = max(1, min(max(args.max_num_seqs, max(levels)), max(levels)))
            os.environ.setdefault("NANOVLLM_CUDAGRAPH_MAX_BS", str(graph_limit))
        vocab_size = get_vocab_size(args.model)
        shared_prefix = make_shared_prefix(args.prefix_len, vocab_size)
        sampling = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            ignore_eos=True,
        )
        warmup = SamplingParams(
            temperature=args.temperature,
            max_tokens=4,
            ignore_eos=True,
        )

        llm = LLM(
            args.model,
            kvcache_type=args.backend,
            tensor_parallel_size=1,
            enforce_eager=args.eager,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=max(args.max_num_seqs, max(levels)),
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        try:
            _ = run_wave(llm, [[1, 2, 3, 4]], warmup)
            _ = run_wave(llm, [shared_prefix], warmup)

            waves: dict[str, dict[str, Any]] = {}
            for concurrency in levels:
                total = WaveMetrics(concurrency=concurrency, rounds=args.rounds)
                for round_idx in range(args.rounds):
                    prompts = make_wave_prompts(
                        concurrency,
                        round_idx,
                        shared_prefix,
                        args.suffix_len,
                        vocab_size,
                    )
                    wave = run_wave(llm, prompts, sampling)
                    merge_wave(total, wave)
                waves[str(concurrency)] = total.to_json()

            peak_allocated, peak_reserved = maybe_cuda_peak_memory()
            return BackendResult(
                ok=True,
                backend=args.backend or "unknown",
                waves=waves,
                peak_memory_allocated_gb=peak_allocated,
                peak_memory_reserved_gb=peak_reserved,
            )
        finally:
            llm.exit()
            del llm
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    except Exception as exc:
        return BackendResult(
            ok=False,
            backend=args.backend or "unknown",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )


def print_worker_result(result: BackendResult) -> None:
    print(f"{RESULT_PREFIX} {json.dumps(asdict(result), sort_keys=True)}", flush=True)


def child_args(args: argparse.Namespace, backend: str) -> list[str]:
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--worker",
        "--backend",
        backend,
        "--model",
        args.model,
        "--hf-model-id",
        args.hf_model_id,
        "--backends",
        args.backends,
        "--concurrency-levels",
        args.concurrency_levels,
        "--rounds",
        str(args.rounds),
        "--prefix-len",
        str(args.prefix_len),
        "--suffix-len",
        str(args.suffix_len),
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        str(args.temperature),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]
    if args.eager:
        cmd.append("--eager")
    return cmd


def run_backend(args: argparse.Namespace, backend: str) -> BackendResult:
    print(f"\n=== Running {backend} ===", flush=True)
    env = os.environ.copy()
    env.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    env.setdefault("NANOVLLM_SAW_INT4_HADAMARD_ORDER", "16")
    proc = subprocess.run(
        child_args(args, backend),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    result_line = None
    passthrough: list[str] = []
    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            result_line = line[len(RESULT_PREFIX) :].strip()
        else:
            passthrough.append(line)
    if passthrough:
        print("\n".join(passthrough[-40:]), flush=True)
    if result_line is None:
        return BackendResult(
            ok=False,
            backend=backend,
            error=f"Worker exited with code {proc.returncode} and emitted no result.",
            traceback=proc.stdout[-4000:],
        )
    return BackendResult(**json.loads(result_line))


def print_summary(results: list[BackendResult], levels: list[int]) -> None:
    print("\n=== Concurrent Cache-Hit Summary ===")
    for result in results:
        if not result.ok:
            print(f"FAILED {result.backend}: {result.error}")

    successes = [r for r in results if r.ok]
    print(
        "backend                 users  wall_s  req/s  out_tok/s  decode_tok/s  "
        "hit_ratio  p50_ms  p95_ms  peak_GB"
    )
    for backend_result in successes:
        for level in levels:
            metrics = backend_result.waves.get(str(level))
            if metrics is None:
                continue
            print(
                f"{backend_result.backend:<23} "
                f"{level:<5}  "
                f"{fmt(metrics.get('wall_time_s')):>7}  "
                f"{fmt(metrics.get('request_per_s'), 2):>5}  "
                f"{fmt(metrics.get('output_tok_s'), 1):>9}  "
                f"{fmt(metrics.get('decode_tok_s'), 1):>12}  "
                f"{fmt(metrics.get('hit_ratio')):>9}  "
                f"{fmt(metrics.get('p50_latency_ms'), 1):>6}  "
                f"{fmt(metrics.get('p95_latency_ms'), 1):>6}  "
                f"{fmt(backend_result.peak_memory_reserved_gb, 2):>7}"
            )

    print(
        "\nRead it as: increasing users should raise throughput until the GPU or "
        "scheduler saturates; p95_ms shows the latency cost of that batching."
    )


def main() -> None:
    args = parse_args()
    if args.worker:
        print_worker_result(run_worker(args))
        return

    args.model = ensure_model(args.model, args.hf_model_id, args.download_if_missing)
    levels = int_csv(args.concurrency_levels)
    backends = csv_list(args.backends)
    if not levels:
        raise SystemExit("No concurrency levels selected.")
    if not backends:
        raise SystemExit("No backends selected.")
    if max(levels) > args.max_num_seqs:
        print("Note: worker will raise max_num_seqs to fit the largest concurrency level.")
    if args.prefix_len % 256:
        print("Warning: prefix_len is not a multiple of 256; nano-vLLM caches full blocks.")

    print("Concurrent workload:")
    print(f"- model={args.model}")
    print(f"- backends={', '.join(backends)}")
    print(f"- concurrency_levels={', '.join(str(x) for x in levels)}, rounds={args.rounds}")
    print(
        f"- prefix_len={args.prefix_len}, suffix_len={args.suffix_len}, "
        f"max_tokens={args.max_tokens}"
    )
    print(f"- decode_mode={'eager' if args.eager else 'cuda_graphs'}")

    results = [run_backend(args, backend) for backend in backends]
    print_summary(results, levels)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, indent=2, sort_keys=True)
        print(f"\nWrote JSON results to {args.json_out}")


if __name__ == "__main__":
    main()
