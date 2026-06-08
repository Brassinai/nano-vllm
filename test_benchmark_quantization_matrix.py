import json
from pathlib import Path
import sys
from types import SimpleNamespace

from safetensors.torch import save_file
import torch

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import benchmark_quantization_matrix as matrix  # noqa: E402


def test_all_model_quantizations_skips_missing_backend_models(tmp_path, capsys):
    dense_model = tmp_path / "dense"
    dense_model.mkdir()
    awq_model = tmp_path / "awq"
    awq_model.mkdir()
    (awq_model / "quantize_config.json").write_text(
        json.dumps(
            {
                "quant_method": "awq",
                "zero_point": True,
                "q_group_size": 128,
                "w_bit": 4,
                "version": "gemm",
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {"model.layers.0.mlp.down_proj.qweight": torch.zeros(1, dtype=torch.int32)},
        awq_model / "model.safetensors",
    )
    args = SimpleNamespace(
        model=str(dense_model),
        model_quantizations="all",
    )

    selected = matrix.filter_missing_quantizations_from_all(
        args,
        ["none", "awq", "gptq"],
        {"awq": str(awq_model)},
    )

    assert selected == ["none", "awq"]
    assert "skipped backends" in capsys.readouterr().out


def test_explicit_missing_quantization_model_still_errors(tmp_path):
    dense_model = tmp_path / "dense"
    dense_model.mkdir()

    try:
        matrix.resolve_candidate_model(
            "gptq",
            str(dense_model),
            {},
            "unused/hf-model",
            False,
        )
    except FileNotFoundError as exc:
        assert "--auto-prepare-gptq" in str(exc)
    else:
        raise AssertionError("Expected missing GPTQ checkpoint to fail.")
