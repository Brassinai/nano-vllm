#!/usr/bin/env python3
"""Benchmark an OpenAI-compatible LLM server (e.g. vLLM started via serve_vllm.sh).

This script mirrors the dataclass/JSON style of the existing nano-vLLM
benchmarks (benchmark_kvcache_backends.py, benchmark_turboquant_compare.py),
but instead of driving the engine directly it issues HTTP requests to a running
``/v1/completions`` endpoint.

Reported metrics per phase:
- Request throughput (req/s)
- Output throughput (output_tok/s)
- Wall time (s)
- TTFT P50/P90/P99 (ms)  -- only valid when --stream is set
- Inter-token latency (ITL) P50/P90/P99 (ms)  -- only valid with --stream
- End-to-end request latency P50/P90/P99 (ms)
- Success / failure counts

The script sends two phases by default:
- ``cache_hit``: every request shares a long prefix (engages prefix caching).
- ``cold``:      each request has its own random prefix (no prefix cache hits).

Designed to be runnable in Google Colab against a vLLM process started in the
background by ``serve_vllm.sh``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any

import aiohttp


RESULT_PREFIX = "RESULT_JSON:"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_MODEL_NAME = "nano-vllm-benchmark"


# ----------------------------------------------------------------------------
# Workload generation (kept consistent with benchmark_kvcache_backends.py so a
# nano-vLLM offline run and a vLLM serving run see comparable token mixes).
# ----------------------------------------------------------------------------
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


def get_vocab_size(model: str | None) -> int:
    if not model:
        return 32000
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model)
        vocab_size = getattr(cfg, "vocab_size", None)
        if isinstance(vocab_size, int) and vocab_size > 8:
            return vocab_size
    except Exception:
        pass
    return 32000


# ----------------------------------------------------------------------------
# Result dataclasses (parallel structure to benchmark_kvcache_backends.py)
# ----------------------------------------------------------------------------
@dataclass
class RequestResult:
    success: bool
    latency_s: float = 0.0
    ttft_ms: float | None = None
    itl_ms: list[float] = field(default_factory=list)
    output_tokens: int = 0
    prompt_tokens: int = 0
    error: str | None = None


@dataclass
class PhaseMetrics:
    phase: str
    num_requests: int = 0
    num_success: int = 0
    num_failed: int = 0
    wall_time_s: float = 0.0
    total_prompt_tokens: int = 0
    total_output_tokens: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    ttfts_ms: list[float] = field(default_factory=list)
    itl_ms: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @staticmethod
    def _percentile(values: list[float], pct: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        if len(ordered) == 1:
            return ordered[0]
        # Same fractional rank used by the existing concurrent benchmark.
        idx = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
        return ordered[idx]

    @property
    def request_throughput(self) -> float:
        return self.num_success / self.wall_time_s if self.wall_time_s > 0 else 0.0

    @property
    def output_throughput(self) -> float:
        return self.total_output_tokens / self.wall_time_s if self.wall_time_s > 0 else 0.0

    @property
    def prompt_throughput(self) -> float:
        return self.total_prompt_tokens / self.wall_time_s if self.wall_time_s > 0 else 0.0

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["request_throughput"] = self.request_throughput
        data["output_throughput"] = self.output_throughput
        data["prompt_throughput"] = self.prompt_throughput
        data["latency_p50_ms"] = self._percentile(self.latencies_ms, 50)
        data["latency_p90_ms"] = self._percentile(self.latencies_ms, 90)
        data["latency_p99_ms"] = self._percentile(self.latencies_ms, 99)
        data["ttft_p50_ms"] = self._percentile(self.ttfts_ms, 50)
        data["ttft_p90_ms"] = self._percentile(self.ttfts_ms, 90)
        data["ttft_p99_ms"] = self._percentile(self.ttfts_ms, 99)
        data["itl_p50_ms"] = self._percentile(self.itl_ms, 50)
        data["itl_p90_ms"] = self._percentile(self.itl_ms, 90)
        data["itl_p99_ms"] = self._percentile(self.itl_ms, 99)
        data["mean_latency_ms"] = (
            statistics.fmean(self.latencies_ms) if self.latencies_ms else None
        )
        # Keep the raw arrays out of the JSON summary: they're useful for
        # ad-hoc analysis but blow up Colab output size. Provide compact copies
        # so callers can opt in to keeping them.
        for key in ("latencies_ms", "ttfts_ms", "itl_ms"):
            data[key + "_count"] = len(data[key])
        return data


@dataclass
class RunResult:
    ok: bool
    base_url: str
    model: str
    phases: dict[str, dict[str, Any]] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    traceback: str | None = None


# ----------------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("BENCHMARK_BASE_URL", DEFAULT_BASE_URL),
        help="OpenAI-compatible server base URL (default: %(default)s).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("BENCHMARK_MODEL", DEFAULT_MODEL_NAME),
        help=(
            "Logical model name the server advertises (--served-model-name in "
            "serve_vllm.sh). Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help=(
            "Optional path/HF id used only to read vocab_size so prompt token "
            "ids match the nano-vLLM offline benchmarks. Falls back to 32000."
        ),
    )
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--prefix-len", type=int, default=512)
    parser.add_argument("--suffix-len", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--stream",
        action="store_true",
        help=(
            "Stream completions so TTFT and inter-token latency can be "
            "measured. Strongly recommended for serving-style benchmarks."
        ),
    )
    parser.add_argument(
        "--phases",
        type=str,
        default="cache_hit,cold",
        help=(
            "Comma-separated phases to run. Supported: cache_hit (shared "
            "prefix) and cold (independent prefixes)."
        ),
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=4,
        help="Number of warmup requests run before each phase (not timed).",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=600.0,
        help="Per-request HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for prompt scrambling (only affects which suffix tokens land where).",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="If provided, write the full RunResult to this path as JSON.",
    )
    return parser.parse_args()


def csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


# ----------------------------------------------------------------------------
# Single-request driver
# ----------------------------------------------------------------------------
async def send_request(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt_token_ids: list[int],
    max_tokens: int,
    temperature: float,
    top_p: float,
    stream: bool,
    timeout: float,
) -> RequestResult:
    url = base_url.rstrip("/") + "/v1/completions"
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt_token_ids,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "ignore_eos": True,
        "stream": stream,
    }
    if stream:
        # Ask the server to include usage in the final SSE frame.
        payload["stream_options"] = {"include_usage": True}

    request_timeout = aiohttp.ClientTimeout(total=timeout)
    t0 = time.perf_counter()
    try:
        async with session.post(url, json=payload, timeout=request_timeout) as resp:
            if resp.status != 200:
                body = (await resp.text())[:500]
                return RequestResult(
                    success=False,
                    latency_s=time.perf_counter() - t0,
                    error=f"HTTP {resp.status}: {body}",
                )

            if not stream:
                data = await resp.json()
                usage = data.get("usage") or {}
                prompt_tokens = int(usage.get("prompt_tokens") or len(prompt_token_ids))
                output_tokens = int(
                    usage.get("completion_tokens")
                    or (usage.get("total_tokens") or 0) - prompt_tokens
                )
                return RequestResult(
                    success=True,
                    latency_s=time.perf_counter() - t0,
                    output_tokens=max(output_tokens, 0),
                    prompt_tokens=prompt_tokens,
                )

            ttft_ms: float | None = None
            itls_ms: list[float] = []
            last_chunk_time = t0
            streamed_chunks = 0
            usage_completion_tokens: int | None = None
            prompt_tokens = len(prompt_token_ids)
            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line or not line.startswith("data:"):
                    continue
                payload_str = line[len("data:"):].strip()
                if payload_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload_str)
                except json.JSONDecodeError:
                    continue

                # Usage frames have no token text but carry final counts.
                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens)
                    completion_tokens = usage.get("completion_tokens")
                    if completion_tokens is not None:
                        usage_completion_tokens = int(completion_tokens)

                choices = chunk.get("choices") or []
                if not choices:
                    continue
                text = choices[0].get("text") or ""
                if not text:
                    continue

                now = time.perf_counter()
                if ttft_ms is None:
                    ttft_ms = (now - t0) * 1000.0
                else:
                    itls_ms.append((now - last_chunk_time) * 1000.0)
                last_chunk_time = now
                streamed_chunks += 1

            # Prefer the server-reported usage; fall back to chunk count if the
            # server did not emit a usage frame (older vLLM, custom backends).
            output_tokens = (
                usage_completion_tokens
                if usage_completion_tokens is not None
                else streamed_chunks
            )

            return RequestResult(
                success=True,
                latency_s=time.perf_counter() - t0,
                ttft_ms=ttft_ms,
                itl_ms=itls_ms,
                output_tokens=output_tokens,
                prompt_tokens=prompt_tokens,
            )
    except asyncio.TimeoutError:
        return RequestResult(
            success=False,
            latency_s=time.perf_counter() - t0,
            error=f"Timeout after {timeout}s",
        )
    except aiohttp.ClientError as exc:
        return RequestResult(
            success=False,
            latency_s=time.perf_counter() - t0,
            error=f"{type(exc).__name__}: {exc}",
        )


# ----------------------------------------------------------------------------
# Phase driver: bounded-concurrency dispatcher
# ----------------------------------------------------------------------------
async def run_phase(
    phase: str,
    prompts: list[list[int]],
    args: argparse.Namespace,
) -> PhaseMetrics:
    metrics = PhaseMetrics(phase=phase, num_requests=len(prompts))
    semaphore = asyncio.Semaphore(args.max_concurrency)
    connector = aiohttp.TCPConnector(limit=args.max_concurrency * 2)
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        async def bounded(prompt: list[int]) -> RequestResult:
            async with semaphore:
                return await send_request(
                    session,
                    args.base_url,
                    args.model,
                    prompt,
                    args.max_tokens,
                    args.temperature,
                    args.top_p,
                    args.stream,
                    args.request_timeout,
                )

        wall_start = time.perf_counter()
        results = await asyncio.gather(*(bounded(p) for p in prompts))
        metrics.wall_time_s = time.perf_counter() - wall_start

    for r in results:
        if r.success:
            metrics.num_success += 1
            metrics.total_prompt_tokens += r.prompt_tokens
            metrics.total_output_tokens += r.output_tokens
            metrics.latencies_ms.append(r.latency_s * 1000.0)
            if r.ttft_ms is not None:
                metrics.ttfts_ms.append(r.ttft_ms)
            metrics.itl_ms.extend(r.itl_ms)
        else:
            metrics.num_failed += 1
            if r.error:
                metrics.errors.append(r.error)
    return metrics


# ----------------------------------------------------------------------------
# Warmup + readiness
# ----------------------------------------------------------------------------
async def wait_for_ready(base_url: str, model: str, timeout_s: float = 60.0) -> bool:
    url = base_url.rstrip("/") + "/v1/models"
    deadline = time.perf_counter() + timeout_s
    async with aiohttp.ClientSession() as session:
        while time.perf_counter() < deadline:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        ids = [
                            entry.get("id")
                            for entry in (data.get("data") or [])
                            if isinstance(entry, dict)
                        ]
                        if model in ids or not ids:
                            return True
            except aiohttp.ClientError:
                pass
            except asyncio.TimeoutError:
                pass
            await asyncio.sleep(1.0)
    return False


async def warmup(args: argparse.Namespace, vocab_size: int) -> None:
    if args.warmup_requests <= 0:
        return
    prompts = make_prompts(
        args.warmup_requests,
        max(args.prefix_len // 4, 8),
        max(args.suffix_len // 4, 4),
        vocab_size,
        shared_prefix=None,
        salt=1234,
    )
    saved_warmup = args.warmup_requests
    args.warmup_requests = 0  # avoid recursion
    try:
        await run_phase("warmup", prompts, args)
    finally:
        args.warmup_requests = saved_warmup


# ----------------------------------------------------------------------------
# Reporting (tabular, matches the formatting of the existing benchmarks)
# ----------------------------------------------------------------------------
def fmt(value: Any, digits: int = 2, na: str = "n/a") -> str:
    if value is None:
        return na
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return na


def print_summary(result: RunResult) -> None:
    print("\n=== Serving Benchmark Summary ===")
    print(f"server: {result.base_url}")
    print(f"model:  {result.model}")
    print(f"config: {json.dumps(result.config, sort_keys=True)}")

    if not result.ok:
        print(f"FAILED: {result.error}")
        return

    print(
        "\nphase       reqs  ok  fail  wall_s  req/s   out_tok/s  prompt_tok/s  "
        "p50_lat_ms  p99_lat_ms  p50_ttft_ms  p99_ttft_ms  p50_itl_ms  p99_itl_ms"
    )
    for phase_name in sorted(result.phases):
        m = result.phases[phase_name]
        print(
            f"{phase_name:<11} "
            f"{m['num_requests']:>4}  "
            f"{m['num_success']:>2}  "
            f"{m['num_failed']:>4}  "
            f"{fmt(m['wall_time_s']):>6}  "
            f"{fmt(m['request_throughput']):>5}  "
            f"{fmt(m['output_throughput'], 1):>9}  "
            f"{fmt(m['prompt_throughput'], 1):>12}  "
            f"{fmt(m.get('latency_p50_ms'), 1):>10}  "
            f"{fmt(m.get('latency_p99_ms'), 1):>10}  "
            f"{fmt(m.get('ttft_p50_ms'), 1):>11}  "
            f"{fmt(m.get('ttft_p99_ms'), 1):>11}  "
            f"{fmt(m.get('itl_p50_ms'), 1):>10}  "
            f"{fmt(m.get('itl_p99_ms'), 1):>10}"
        )

    print(
        "\nReading guide: higher req/s and out_tok/s are better. "
        "TTFT measures the time to the first streamed token; ITL is the inter-"
        "token latency. Both require --stream to be set."
    )


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
async def amain(args: argparse.Namespace) -> RunResult:
    random.seed(args.seed)
    vocab_size = get_vocab_size(args.tokenizer or args.model)

    config = {
        "num_prompts": args.num_prompts,
        "max_concurrency": args.max_concurrency,
        "prefix_len": args.prefix_len,
        "suffix_len": args.suffix_len,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "stream": args.stream,
        "vocab_size": vocab_size,
        "warmup_requests": args.warmup_requests,
        "seed": args.seed,
    }

    ready = await wait_for_ready(args.base_url, args.model, timeout_s=60.0)
    if not ready:
        return RunResult(
            ok=False,
            base_url=args.base_url,
            model=args.model,
            config=config,
            error=(
                "Server did not advertise the requested model within 60s. "
                "Check that serve_vllm.sh has finished starting and that "
                "--model / --served-model-name agree."
            ),
        )

    await warmup(args, vocab_size)

    phases = csv_list(args.phases)
    unknown = sorted(set(phases) - {"cache_hit", "cold"})
    if unknown:
        return RunResult(
            ok=False,
            base_url=args.base_url,
            model=args.model,
            config=config,
            error=f"Unknown phases requested: {unknown}",
        )

    shared_prefix = make_shared_prefix(args.prefix_len, vocab_size)
    phase_results: dict[str, dict[str, Any]] = {}
    for phase in phases:
        if phase == "cache_hit":
            prompts = make_prompts(
                args.num_prompts,
                args.prefix_len,
                args.suffix_len,
                vocab_size,
                shared_prefix=shared_prefix,
            )
        else:  # cold
            prompts = make_prompts(
                args.num_prompts,
                args.prefix_len,
                args.suffix_len,
                vocab_size,
                shared_prefix=None,
                salt=8192 + args.seed,
            )
        # Issue a single seed request so the server has the prefix cached
        # before the timed cache_hit phase. This matches the seeding pattern
        # used by benchmark_kvcache_backends.py.
        if phase == "cache_hit":
            seed_prompt = make_prompts(
                1,
                args.prefix_len,
                args.suffix_len,
                vocab_size,
                shared_prefix=shared_prefix,
            )
            seeding_args = argparse.Namespace(**vars(args))
            seeding_args.max_tokens = max(args.max_tokens // 16, 4)
            await run_phase("seed", seed_prompt, seeding_args)

        print(f"\n=== Phase: {phase} ===")
        metrics = await run_phase(phase, prompts, args)
        phase_results[phase] = metrics.to_json()
        print(
            f"  finished {metrics.num_success}/{metrics.num_requests} in "
            f"{metrics.wall_time_s:.2f}s (failures: {metrics.num_failed})"
        )

    return RunResult(
        ok=True,
        base_url=args.base_url,
        model=args.model,
        phases=phase_results,
        config=config,
    )


def main() -> None:
    args = parse_args()
    try:
        result = asyncio.run(amain(args))
    except Exception as exc:
        result = RunResult(
            ok=False,
            base_url=args.base_url,
            model=args.model,
            config={},
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )

    print_summary(result)
    # Emit a machine-readable JSON line for orchestration scripts that want
    # to pipe stdout (same pattern used by the other benchmarks).
    print(f"{RESULT_PREFIX} {json.dumps(asdict(result), sort_keys=True)}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2, sort_keys=True)
        print(f"Wrote JSON results to {args.json_out}")

    sys.exit(0 if result.ok else 1)


if __name__ == "__main__":
    main()
