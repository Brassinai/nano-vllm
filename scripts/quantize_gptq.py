#!/usr/bin/env python3
"""Quantize a dense local Hugging Face model into nano-vLLM GPTQ format."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from nanovllm.quantization.gptq_export import (
    DEFAULT_CALIBRATION_TEXTS,
    GPTQExportConfig,
    export_gptq_checkpoint,
)


def default_output_path(model: str, bits: int, group_size: int) -> str:
    src = Path(os.path.expanduser(model)).resolve()
    return str(src.parent / f"{src.name}-gptq-w{bits}-g{group_size}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to a dense local Hugging Face model directory.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Destination directory for the GPTQ checkpoint.",
    )
    parser.add_argument("--bits", type=int, default=4, choices=(2, 4, 8))
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--nsamples", type=int, default=32)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16" if torch.cuda.is_available() else "float32",
        choices=("float16", "float32"),
    )
    parser.add_argument(
        "--calibration-file",
        type=str,
        default=None,
        help="Optional text file with one calibration sample per line.",
    )
    parser.add_argument(
        "--calibration-text",
        action="append",
        default=[],
        help="Additional calibration text. Pass multiple times for more samples.",
    )
    return parser.parse_args()


def load_calibration_texts(args: argparse.Namespace) -> list[str]:
    texts = list(args.calibration_text)
    if args.calibration_file:
        with open(os.path.expanduser(args.calibration_file), encoding="utf-8") as f:
            texts.extend(line.strip() for line in f if line.strip())
    if texts:
        return texts
    return list(DEFAULT_CALIBRATION_TEXTS)


def main() -> None:
    args = parse_args()
    output = args.output or default_output_path(args.model, args.bits, args.group_size)
    if args.group_size <= 0 and args.group_size != -1:
        raise SystemExit("--group-size must be positive or -1.")

    config = GPTQExportConfig(
        bits=args.bits,
        group_size=args.group_size,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        seed=args.seed,
        blocksize=args.blocksize,
        percdamp=args.percdamp,
        dtype=args.dtype,
        device=args.device,
    )
    texts = load_calibration_texts(args)
    result = export_gptq_checkpoint(
        args.model,
        output,
        config=config,
        calibration_texts=texts,
    )
    print(result)


if __name__ == "__main__":
    main()
