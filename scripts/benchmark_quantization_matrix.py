#!/usr/bin/env python3
"""Benchmark model-weight quantization against KV-cache quantization.

This script sweeps a matrix of:
- model-weight quantization backends such as ``none`` and ``gptq``
- KV-cache backends such as ``default``, ``int8``, ``int4``, and TurboQuant

Each candidate runs in a fresh subprocess so CUDA memory, NCCL state, and JIT
artifacts do not leak across runs.

Examples:

Benchmark the dense baseline and GPTQ against the default cache plus all
canonical KV-cache quantizers:

    python3 scripts/benchmark_quantization_matrix.py \
      --model ~/huggingface/Qwen3-0.6B \
      --model-quantizations none,gptq \
      --quantization-models gptq=~/huggingface/Qwen3-0.6B-GPTQ

Ask the benchmark to include every registered model quantization backend:

    python3 scripts/benchmark_quantization_matrix.py \
      --model ~/huggingface/Qwen3-0.6B \
      --model-quantizations all \
      --quantization-models gptq=~/huggingface/Qwen3-0.6B-GPTQ
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
from typing import Any

from benchmark_kvcache_backends import (
    RESULT_PREFIX,
    cache_memory_stats,
    csv_list,
    default_model_path,
    ensure_model,
    fmt,
    get_vocab_size,
    make_prompts,
    make_shared_prefix,
    maybe_cuda_peak_memory,
    run_requests,
)

from nanovllm.kvcache import KVCacheRegistry
from nanovllm.quantization import QuantizationRegistry


KV_ALIAS_EQUIVALENTS = {
    "turboquant": "turboquant_k4v4",
    "int4_bdr": "saw_int4",
}
CANONICAL_KV_ORDER = (
    "default",
    "int8",
    "int4",
    "saw_int4",
    "turboquant_k4v4",
    "turboquant_k3v4",
    "turboquant_k3v3",
)
DEFAULT_MODEL_QUANTIZATIONS = "none,gptq"
DEFAULT_KV_BACKENDS = ",".join(CANONICAL_KV_ORDER)


@dataclass
class MatrixResult:
    ok: bool
    model_quantization: str
    kv_cache_type: str
    model_path: str
    phases: dict[str, dict[str, Any]] = field(default_factory=dict)
    peak_memory_allocated_gb: float | None = None
    peak_memory_reserved_gb: float | None = None
    cache_memory_gb: float | None = None
    cache_bytes_per_token: float | None = None
    num_kvcache_blocks: int | None = None
    error: str | None = None
    traceback: str | None = None

    @property
    def label(self) -> str:
        return f"{self.model_quantization}+{self.kv_cache_type}"


def normalize_model_quantization(name: str) -> str:
    key = name.strip().lower()
    aliases = {
        "": "none",
        "fp16": "none",
        "dense": "none",
        "baseline": "none",
        "unquantized": "none",
    }
    return aliases.get(key, key)


def available_model_quantizations() -> list[str]:
    return ["none", *QuantizationRegistry.list_backends()]


def available_kv_backends() -> list[str]:
    registered = set(KVCacheRegistry.list_caches())
    ordered = [name for name in CANONICAL_KV_ORDER if name in registered]
    extras = sorted(
        name
        for name in registered
        if name not in ordered and name not in KV_ALIAS_EQUIVALENTS
    )
    return ordered + extras


def expand_model_quantizations(raw: str) -> list[str]:
    requested = csv_list(raw)
    if not requested:
        raise ValueError("No model quantization backends selected.")

    available = set(available_model_quantizations())
    expanded: list[str] = []
    for item in requested:
        normalized = normalize_model_quantization(item)
        if normalized == "all":
            for name in available_model_quantizations():
                if name not in expanded:
                    expanded.append(name)
            continue
        if normalized not in available:
            raise ValueError(
                f"Unknown model quantization backend {item!r}. "
                f"Available: {sorted(available)}"
            )
        if normalized not in expanded:
            expanded.append(normalized)
    return expanded


def expand_kv_backends(raw: str) -> list[str]:
    requested = csv_list(raw)
    if not requested:
        raise ValueError("No KV-cache backends selected.")

    available = set(KVCacheRegistry.list_caches())
    expanded: list[str] = []
    for item in requested:
        normalized = KV_ALIAS_EQUIVALENTS.get(item.strip(), item.strip())
        if normalized == "all":
            for name in available_kv_backends():
                if name not in expanded:
                    expanded.append(name)
            continue
        if normalized not in available:
            raise ValueError(
                f"Unknown KV-cache backend {item!r}. "
                f"Available: {sorted(available)}"
            )
        if normalized not in expanded:
            expanded.append(normalized)
    return expanded


def parse_quantization_models(raw: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not raw.strip():
        return mapping
    for item in csv_list(raw):
        if "=" not in item:
            raise ValueError(
                "Expected --quantization-models entries in the form "
                "'backend=/path/to/model'."
            )
        key, value = item.split("=", 1)
        normalized = normalize_model_quantization(key)
        if not value.strip():
            raise ValueError(f"Missing model path for quantization {key!r}.")
        mapping[normalized] = value.strip()
    return mapping


def resolve_candidate_model(
    model_quantization: str,
    fallback_model: str,
    quantization_models: dict[str, str],
    hf_model_id: str,
    download_if_missing: bool,
) -> str:
    chosen = os.path.expanduser(quantization_models.get(model_quantization, fallback_model))
    path = Path(chosen)

    # Only the shared fallback model can be auto-downloaded. Backend-specific
    # checkpoints are expected to already exist locally.
    if chosen == os.path.expanduser(fallback_model):
        return ensure_model(chosen, hf_model_id, download_if_missing)

    if not path.is_dir():
        raise FileNotFoundError(
            f"Model directory for quantization {model_quantization!r} was not found: "
            f"{path}. Pass --quantization-models {model_quantization}=/path/to/model."
        )
    return str(path.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default=default_model_path())
    parser.add_argument("--hf-model-id", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--download-if-missing", action="store_true")
    parser.add_argument(
        "--model-quantizations",
        type=str,
        default=DEFAULT_MODEL_QUANTIZATIONS,
        help=(
            "Comma-separated model-weight quantization backends. "
            "Use 'none' for the dense baseline or 'all' to include every "
            f"registered backend. Default: {DEFAULT_MODEL_QUANTIZATIONS}"
        ),
    )
    parser.add_argument(
        "--quantization-models",
        type=str,
        default="",
        help=(
            "Optional backend-specific local model directories in the form "
            "'gptq=/path/to/gptq-model,none=/path/to/base-model'. Backends "
            "without an explicit entry fall back to --model."
        ),
    )
    parser.add_argument(
        "--kv-backends",
        type=str,
        default=DEFAULT_KV_BACKENDS,
        help=(
            "Comma-separated KV-cache backends. Use 'all' to include every "
            f"canonical registered backend. Default: {DEFAULT_KV_BACKENDS}"
        ),
    )
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
        "--num-kvcache-blocks",
        type=int,
        default=-1,
        help=(
            "Use a fixed KV-cache block count when comparing storage footprints. "
            "The default auto mode lets denser backends allocate more blocks."
        ),
    )
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
    parser.add_argument(
        "--worker-model-quantization",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-kv-backend",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-model-path",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def run_worker(args: argparse.Namespace) -> MatrixResult:
    try:
        import torch
        from nanovllm import LLM, SamplingParams

        os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
        os.environ.setdefault("NANOVLLM_SAW_INT4_HADAMARD_ORDER", "16")
        if not args.eager:
            graph_limit = max(1, min(args.max_num_seqs, args.num_prompts))
            os.environ.setdefault("NANOVLLM_CUDAGRAPH_MAX_BS", str(graph_limit))

        model_quantization = normalize_model_quantization(
            args.worker_model_quantization or "none"
        )
        kv_backend = args.worker_kv_backend or "default"
        model_path = args.worker_model_path or args.model

        vocab_size = get_vocab_size(model_path)
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
            model_path,
            quantization=None if model_quantization == "none" else model_quantization,
            kvcache_type=kv_backend,
            tensor_parallel_size=1,
            enforce_eager=args.eager,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
            num_kvcache_blocks=args.num_kvcache_blocks,
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
            cache_gb, cache_bytes_per_token, num_kvcache_blocks = cache_memory_stats(llm)
            return MatrixResult(
                ok=True,
                model_quantization=model_quantization,
                kv_cache_type=kv_backend,
                model_path=model_path,
                phases=phases,
                peak_memory_allocated_gb=peak_allocated,
                peak_memory_reserved_gb=peak_reserved,
                cache_memory_gb=cache_gb,
                cache_bytes_per_token=cache_bytes_per_token,
                num_kvcache_blocks=num_kvcache_blocks,
            )
        finally:
            llm.exit()
            del llm
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    except Exception as exc:
        return MatrixResult(
            ok=False,
            model_quantization=normalize_model_quantization(
                args.worker_model_quantization or "none"
            ),
            kv_cache_type=args.worker_kv_backend or "default",
            model_path=args.worker_model_path or args.model,
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )


def print_worker_result(result: MatrixResult) -> None:
    print(f"{RESULT_PREFIX} {json.dumps(asdict(result), sort_keys=True)}", flush=True)


def child_args(
    args: argparse.Namespace,
    model_quantization: str,
    kv_backend: str,
    model_path: str,
) -> list[str]:
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--worker",
        "--worker-model-quantization",
        model_quantization,
        "--worker-kv-backend",
        kv_backend,
        "--worker-model-path",
        model_path,
        "--model",
        args.model,
        "--hf-model-id",
        args.hf_model_id,
        "--model-quantizations",
        args.model_quantizations,
        "--quantization-models",
        args.quantization_models,
        "--kv-backends",
        args.kv_backends,
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
        "--num-kvcache-blocks",
        str(args.num_kvcache_blocks),
    ]
    if args.eager:
        cmd.append("--eager")
    if args.skip_cold:
        cmd.append("--skip-cold")
    return cmd


def run_candidate(
    args: argparse.Namespace,
    model_quantization: str,
    kv_backend: str,
    model_path: str,
) -> MatrixResult:
    label = f"{model_quantization}+{kv_backend}"
    print(f"\n=== Running {label} ===", flush=True)
    env = os.environ.copy()
    env.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    env.setdefault("NANOVLLM_SAW_INT4_HADAMARD_ORDER", "16")
    proc = subprocess.run(
        child_args(args, model_quantization, kv_backend, model_path),
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
        return MatrixResult(
            ok=False,
            model_quantization=model_quantization,
            kv_cache_type=kv_backend,
            model_path=model_path,
            error=f"Worker exited with code {proc.returncode} and emitted no result.",
            traceback=proc.stdout[-4000:],
        )
    return MatrixResult(**json.loads(result_line))


def print_summary(results: list[MatrixResult]) -> None:
    print("\n=== Summary ===")
    failures = [r for r in results if not r.ok]
    for failed in failures:
        print(f"FAILED {failed.label}: {failed.error}")

    successes = [r for r in results if r.ok]
    phases = sorted({phase for r in successes for phase in r.phases})
    for phase in phases:
        rows = [
            (r.phases[phase].get("wall_time_s", float("inf")), r, r.phases[phase])
            for r in successes
            if phase in r.phases
        ]
        rows.sort(key=lambda item: item[0])

        baseline_same_kv = {
            r.kv_cache_type: metrics
            for _, r, metrics in rows
            if r.model_quantization == "none"
        }
        baseline_same_model = {
            r.model_quantization: metrics
            for _, r, metrics in rows
            if r.kv_cache_type == "default"
        }

        print(f"\nPhase: {phase}")
        print(
            "rank  model_quant   kv_cache           wall_s  vs_kv_none  "
            "vs_model_default  prefill_tok/s  decode_tok/s  e2e_out_tok/s  "
            "hit_ratio  blocks  B/tok/l  cache_GB  peak_GB"
        )
        for rank, (_, result, metrics) in enumerate(rows, start=1):
            wall = metrics.get("wall_time_s")
            kv_baseline = baseline_same_kv.get(result.kv_cache_type, {})
            model_baseline = baseline_same_model.get(result.model_quantization, {})
            speedup_vs_kv_none = (
                kv_baseline.get("wall_time_s") / wall
                if kv_baseline.get("wall_time_s") and wall
                else None
            )
            speedup_vs_model_default = (
                model_baseline.get("wall_time_s") / wall
                if model_baseline.get("wall_time_s") and wall
                else None
            )
            print(
                f"{rank:<5} "
                f"{result.model_quantization:<12} "
                f"{result.kv_cache_type:<17} "
                f"{fmt(wall):>7}  "
                f"{fmt(speedup_vs_kv_none, 2):>10}  "
                f"{fmt(speedup_vs_model_default, 2):>16}  "
                f"{fmt(metrics.get('prefill_tps'), 1):>13}  "
                f"{fmt(metrics.get('decode_tps'), 1):>12}  "
                f"{fmt(metrics.get('end_to_end_tps'), 1):>13}  "
                f"{fmt(metrics.get('hit_ratio')):>9}  "
                f"{result.num_kvcache_blocks if result.num_kvcache_blocks is not None else 'n/a':>6}  "
                f"{fmt(result.cache_bytes_per_token, 0):>7}  "
                f"{fmt(result.cache_memory_gb, 2):>8}  "
                f"{fmt(result.peak_memory_reserved_gb, 2):>7}"
            )

    print(
        "\nRead it as: vs_kv_none compares a quantized model against the dense "
        "model on the same KV-cache backend, while vs_model_default compares a "
        "quantized KV cache against the default cache for the same model-weight "
        "quantization. Lower wall_s is better and higher tok/s is better."
    )


def main() -> None:
    args = parse_args()
    if args.worker:
        print_worker_result(run_worker(args))
        return

    model_quantizations = expand_model_quantizations(args.model_quantizations)
    kv_backends = expand_kv_backends(args.kv_backends)
    quantization_models = parse_quantization_models(args.quantization_models)
    resolved_models = {
        model_quantization: resolve_candidate_model(
            model_quantization,
            args.model,
            quantization_models,
            args.hf_model_id,
            args.download_if_missing,
        )
        for model_quantization in model_quantizations
    }

    if args.prefix_len % 256:
        print("Warning: prefix_len is not a multiple of 256; nano-vLLM caches full blocks.")

    print("Benchmark workload:")
    print(f"- model_quantizations={', '.join(model_quantizations)}")
    print(f"- kv_backends={', '.join(kv_backends)}")
    print(
        f"- prompts={args.num_prompts}, prefix_len={args.prefix_len}, "
        f"suffix_len={args.suffix_len}, max_tokens={args.max_tokens}"
    )
    print(f"- decode_mode={'eager' if args.eager else 'cuda_graphs'}")
    print("- resolved_models:")
    for model_quantization in model_quantizations:
        print(f"  - {model_quantization}: {resolved_models[model_quantization]}")

    results: list[MatrixResult] = []
    for model_quantization in model_quantizations:
        for kv_backend in kv_backends:
            results.append(
                run_candidate(
                    args,
                    model_quantization,
                    kv_backend,
                    resolved_models[model_quantization],
                )
            )

    print_summary(results)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, indent=2, sort_keys=True)
        print(f"\nWrote JSON results to {args.json_out}")


if __name__ == "__main__":
    main()
