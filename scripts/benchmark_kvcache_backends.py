#!/usr/bin/env python3
"""Benchmark nano-vLLM KV-cache backends with cache-hit and cold prompts.

The default candidate list compares:
- default FP16/BF16 KV cache
- TurboQuant 4/4, 3/4, and 3/3 bit presets
- SAW-INT4

Each backend runs in a fresh subprocess so CUDA memory and JIT state do not
leak between candidates. Prompt token IDs intentionally share a full-block
prefix in the cache-hit phase.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any


RESULT_PREFIX = "RESULT_JSON:"
DEFAULT_BACKENDS = "default,turboquant_k4v4,turboquant_k3v4,turboquant_k3v3,saw_int4"


def default_model_path() -> str:
    if Path("/content").exists():
        return "/content/models/Qwen3-0.6B"
    return os.path.expanduser("~/huggingface/Qwen3-0.6B")


@dataclass
class PhaseMetrics:
    wall_time_s: float = 0.0
    prefill_time_s: float = 0.0
    decode_time_s: float = 0.0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    compute_prompt_tokens: int = 0
    generated_tokens: int = 0

    @property
    def hit_ratio(self) -> float:
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0

    @property
    def prefill_tps(self) -> float:
        return (
            self.compute_prompt_tokens / self.prefill_time_s
            if self.prefill_time_s > 0
            else 0.0
        )

    @property
    def decode_tps(self) -> float:
        return (
            self.generated_tokens / self.decode_time_s
            if self.decode_time_s > 0
            else 0.0
        )

    @property
    def end_to_end_tps(self) -> float:
        return (
            self.generated_tokens / self.wall_time_s
            if self.wall_time_s > 0
            else 0.0
        )

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["hit_ratio"] = self.hit_ratio
        data["prefill_tps"] = self.prefill_tps
        data["decode_tps"] = self.decode_tps
        data["end_to_end_tps"] = self.end_to_end_tps
        return data


@dataclass
class CandidateResult:
    ok: bool
    backend: str
    phases: dict[str, dict[str, Any]] = field(default_factory=dict)
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
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--prefix-len", type=int, default=512)
    parser.add_argument("--suffix-len", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--seed-max-tokens", type=int, default=4)
    parser.add_argument("--warmup-tokens", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
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
    parser.add_argument("--skip-cold", action="store_true")
    parser.add_argument("--json-out", type=str, default=None)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--backend", type=str, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def ensure_model(model: str, hf_model_id: str, download_if_missing: bool) -> str:
    path = Path(os.path.expanduser(model))
    if path.is_dir():
        return str(path.resolve())
    if not download_if_missing:
        raise SystemExit(
            f"Model directory not found: {path}. Pass --download-if-missing on Colab."
        )

    print(f"Downloading {hf_model_id} to {path} ...", flush=True)
    from huggingface_hub import snapshot_download

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=hf_model_id,
            local_dir=str(path),
            local_dir_use_symlinks=False,
        )
    except TypeError:
        snapshot_download(repo_id=hf_model_id, local_dir=str(path))
    return str(path.resolve())


def get_vocab_size(model: str) -> int:
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model)
        vocab_size = getattr(cfg, "vocab_size", None)
        if isinstance(vocab_size, int) and vocab_size > 8:
            return vocab_size
    except Exception:
        pass
    return 32000


def token_id(base: int, offset: int, vocab_size: int) -> int:
    usable = max(vocab_size - 1, 1)
    return 1 + ((base + offset) % usable)


def make_shared_prefix(prefix_len: int, vocab_size: int) -> list[int]:
    return [token_id(100, i, vocab_size) for i in range(prefix_len)]


def make_prompts(
    num_prompts: int,
    prefix_len: int,
    suffix_len: int,
    vocab_size: int,
    *,
    shared_prefix: list[int] | None,
    salt: int = 0,
) -> list[list[int]]:
    prompts: list[list[int]] = []
    for req_idx in range(num_prompts):
        if shared_prefix is None:
            prefix = [
                token_id(300 + req_idx * 997 + salt, i, vocab_size)
                for i in range(prefix_len)
            ]
        else:
            prefix = shared_prefix
        suffix = [
            token_id(20_000 + req_idx * 1009 + salt, i, vocab_size)
            for i in range(suffix_len)
        ]
        prompts.append(prefix + suffix)
    return prompts


def cuda_sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def maybe_cuda_peak_memory() -> tuple[float | None, float | None]:
    try:
        import torch

        if not torch.cuda.is_available():
            return None, None
        return (
            torch.cuda.max_memory_allocated() / (1024**3),
            torch.cuda.max_memory_reserved() / (1024**3),
        )
    except Exception:
        return None, None


def run_requests(llm: Any, prompts: list[list[int]], sampling_params: Any) -> PhaseMetrics:
    for prompt in prompts:
        llm.add_request(prompt, sampling_params)

    metrics = PhaseMetrics()
    wall_start = perf_counter()
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
        if is_prefill:
            metrics.prompt_tokens += prompt_tokens
            metrics.cached_tokens += cached_tokens
            metrics.compute_prompt_tokens += compute_tokens
            metrics.prefill_time_s += step_time
        else:
            metrics.generated_tokens += len(seqs)
            metrics.decode_time_s += step_time

    metrics.wall_time_s = perf_counter() - wall_start
    return metrics


def run_worker(args: argparse.Namespace) -> CandidateResult:
    try:
        import torch
        from nanovllm import LLM, SamplingParams

        os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
        os.environ.setdefault("NANOVLLM_SAW_INT4_HADAMARD_ORDER", "16")
        if not args.eager:
            graph_limit = max(1, min(args.max_num_seqs, args.num_prompts))
            os.environ.setdefault("NANOVLLM_CUDAGRAPH_MAX_BS", str(graph_limit))

        vocab_size = get_vocab_size(args.model)
        shared_prefix = make_shared_prefix(args.prefix_len, vocab_size)
        seed_prompt = [shared_prefix]
        hit_prompts = make_prompts(
            args.num_prompts,
            args.prefix_len,
            args.suffix_len,
            vocab_size,
            shared_prefix=shared_prefix,
        )
        cold_prompts = make_prompts(
            args.num_prompts,
            args.prefix_len,
            args.suffix_len,
            vocab_size,
            shared_prefix=None,
            salt=8192,
        )

        llm = LLM(
            args.model,
            kvcache_type=args.backend,
            tensor_parallel_size=1,
            enforce_eager=args.eager,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        try:
            warmup_sp = SamplingParams(
                temperature=args.temperature,
                max_tokens=args.warmup_tokens,
                ignore_eos=True,
            )
            seed_sp = SamplingParams(
                temperature=args.temperature,
                max_tokens=args.seed_max_tokens,
                ignore_eos=True,
            )
            bench_sp = SamplingParams(
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                ignore_eos=True,
            )
            _ = run_requests(llm, [[1, 2, 3, 4]], warmup_sp)
            _ = run_requests(llm, seed_prompt, seed_sp)

            phases = {"cache_hit": run_requests(llm, hit_prompts, bench_sp).to_json()}
            if not args.skip_cold:
                phases["cold"] = run_requests(llm, cold_prompts, bench_sp).to_json()

            peak_allocated, peak_reserved = maybe_cuda_peak_memory()
            return CandidateResult(
                ok=True,
                backend=args.backend or "unknown",
                phases=phases,
                peak_memory_allocated_gb=peak_allocated,
                peak_memory_reserved_gb=peak_reserved,
            )
        finally:
            llm.exit()
            del llm
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    except Exception as exc:
        return CandidateResult(
            ok=False,
            backend=args.backend or "unknown",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )


def print_worker_result(result: CandidateResult) -> None:
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
        "--num-prompts",
        str(args.num_prompts),
        "--prefix-len",
        str(args.prefix_len),
        "--suffix-len",
        str(args.suffix_len),
        "--max-tokens",
        str(args.max_tokens),
        "--seed-max-tokens",
        str(args.seed_max_tokens),
        "--warmup-tokens",
        str(args.warmup_tokens),
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
    if args.skip_cold:
        cmd.append("--skip-cold")
    return cmd


def run_candidate(args: argparse.Namespace, backend: str) -> CandidateResult:
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
        return CandidateResult(
            ok=False,
            backend=backend,
            error=f"Worker exited with code {proc.returncode} and emitted no result.",
            traceback=proc.stdout[-4000:],
        )
    return CandidateResult(**json.loads(result_line))


def fmt(value: Any, digits: int = 3, na: str = "n/a") -> str:
    if value is None:
        return na
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return na


def print_summary(results: list[CandidateResult]) -> None:
    print("\n=== Summary ===")
    failures = [r for r in results if not r.ok]
    for failed in failures:
        print(f"FAILED {failed.backend}: {failed.error}")

    successes = [r for r in results if r.ok]
    phases = sorted({phase for r in successes for phase in r.phases})
    for phase in phases:
        rows = [
            (r.phases[phase].get("wall_time_s", float("inf")), r, r.phases[phase])
            for r in successes
            if phase in r.phases
        ]
        rows.sort(key=lambda item: item[0])
        baseline = next((m for _, r, m in rows if r.backend == "default"), None)
        base_wall = baseline.get("wall_time_s") if baseline else None

        print(f"\nPhase: {phase}")
        print(
            "rank  backend                 wall_s  speedup  prefill_tok/s  "
            "decode_tok/s  e2e_out_tok/s  hit_ratio  peak_GB"
        )
        for rank, (_, result, metrics) in enumerate(rows, start=1):
            wall = metrics.get("wall_time_s")
            speedup = base_wall / wall if base_wall and wall else None
            print(
                f"{rank:<5} "
                f"{result.backend:<23} "
                f"{fmt(wall):>7}  "
                f"{fmt(speedup, 2):>7}  "
                f"{fmt(metrics.get('prefill_tps'), 1):>13}  "
                f"{fmt(metrics.get('decode_tps'), 1):>12}  "
                f"{fmt(metrics.get('end_to_end_tps'), 1):>13}  "
                f"{fmt(metrics.get('hit_ratio')):>9}  "
                f"{fmt(result.peak_memory_reserved_gb, 2):>7}"
            )

    print(
        "\nRead it as: lower wall_s is better, higher tok/s is better, and "
        "cache_hit hit_ratio should be close to prefix_len / (prefix_len + suffix_len)."
    )


def main() -> None:
    args = parse_args()
    if args.worker:
        print_worker_result(run_worker(args))
        return

    args.model = ensure_model(args.model, args.hf_model_id, args.download_if_missing)
    if args.prefix_len % 256:
        print("Warning: prefix_len is not a multiple of 256; nano-vLLM caches full blocks.")
    backends = csv_list(args.backends)
    if not backends:
        raise SystemExit("No backends selected.")

    print("Benchmark workload:")
    print(f"- model={args.model}")
    print(f"- backends={', '.join(backends)}")
    print(
        f"- prompts={args.num_prompts}, prefix_len={args.prefix_len}, "
        f"suffix_len={args.suffix_len}, max_tokens={args.max_tokens}"
    )
    print(f"- decode_mode={'eager' if args.eager else 'cuda_graphs'}")

    results = [run_candidate(args, backend) for backend in backends]
    print_summary(results)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, indent=2, sort_keys=True)
        print(f"\nWrote JSON results to {args.json_out}")


if __name__ == "__main__":
    main()
