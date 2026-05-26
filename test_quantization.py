import json
from types import SimpleNamespace

import torch
from torch import nn

import pytest
from safetensors.torch import save_file
from transformers import Qwen3Config

import nanovllm.models.qwen3  # noqa: F401
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.quantization import QuantizationRegistry
from nanovllm.quantization.gptq import GPTQConfig, GPTQLinearMethod


class FakeLinear(nn.Module):
    supports_weight_quantization = True

    def __init__(self):
        super().__init__()
        self.input_size = 64
        self.output_size = 128
        self.full_input_size = 64
        self.full_output_size = 128
        self.input_partition_start = 0


def test_gptq_backend_is_registered():
    assert "gptq" in QuantizationRegistry.list_backends()


def test_quantization_registry_rejects_duplicate_backend_names():
    with pytest.raises(ValueError, match="already registered"):
        @QuantizationRegistry.register("gptq")
        class DuplicateGPTQConfig(GPTQConfig):
            pass


def test_gptq_config_parses_checkpoint_metadata():
    quant_config = GPTQConfig.from_config(
        {
            "bits": 4,
            "group_size": 32,
            "desc_act": False,
            "checkpoint_format": "gptq_v2",
        }
    )

    assert isinstance(quant_config, GPTQConfig)
    assert quant_config.pack_factor == 8
    assert quant_config.zero_offset == 0


def test_gptq_rejects_activation_order_until_supported():
    with pytest.raises(ValueError, match="desc_act"):
        GPTQConfig(bits=4, group_size=32, desc_act=True)


def test_gptq_linear_method_creates_packed_checkpoint_parameters():
    layer = FakeLinear()
    quant_config = GPTQConfig(bits=4, group_size=32, desc_act=False)

    method = quant_config.get_quant_method(layer, "fake.linear")

    assert isinstance(method, GPTQLinearMethod)
    method.create_weights(layer)
    assert layer.qweight.shape == (8, 128)
    assert layer.qweight.dtype == torch.int32
    assert layer.qzeros.shape == (2, 16)
    assert layer.scales.shape == (2, 128)
    assert layer.g_idx.tolist() == [0] * 32 + [1] * 32


def test_gptq_fused_module_selection_requires_all_checkpoint_shards():
    quant_config = GPTQConfig(bits=4, group_size=32, desc_act=False)
    quant_config.quantized_modules = {
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
    }

    assert quant_config._is_layer_quantized(
        "model.layers.0.self_attn.qkv_proj"
    )
    assert not quant_config._is_layer_quantized(
        "model.layers.0.self_attn.o_proj"
    )

    quant_config.quantized_modules.remove("model.layers.0.self_attn.v_proj")
    with pytest.raises(ValueError, match="part of fused layer"):
        quant_config._is_layer_quantized("model.layers.0.self_attn.qkv_proj")


def test_model_runner_applies_gptq_quantization_to_engine_model(monkeypatch, tmp_path):
    quantized_modules = {
        "model.layers.0.self_attn.qkv_proj.qweight": torch.zeros(1, dtype=torch.int32),
        "model.layers.0.mlp.gate_up_proj.qweight": torch.zeros(1, dtype=torch.int32),
    }
    save_file(quantized_modules, tmp_path / "model.safetensors")
    (tmp_path / "quantize_config.json").write_text(
        json.dumps(
            {
                "bits": 4,
                "group_size": 32,
                "desc_act": False,
                "checkpoint_format": "gptq_v2",
            }
        ),
        encoding="utf-8",
    )

    hf_config = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        hidden_act="silu",
        tie_word_embeddings=False,
    )
    hf_config.architectures = ["Qwen3ForCausalLM"]
    hf_config.torch_dtype = torch.float16

    config = SimpleNamespace(
        model=str(tmp_path),
        hf_config=hf_config,
        kvcache_block_size=256,
        enforce_eager=True,
        tensor_parallel_size=1,
        quantization="gptq",
        model_architecture="qwen3",
        kvcache_type="default",
    )

    class DummyAttentionBackend:
        name = "dummy"
        supports_cudagraph_capture = True

    monkeypatch.setattr("torch.distributed.init_process_group", lambda *args, **kwargs: None)
    monkeypatch.setattr("torch.distributed.get_rank", lambda: 0)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda: 1)
    monkeypatch.setattr("torch.cuda.set_device", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch, "set_default_device", lambda *args, **kwargs: None)
    monkeypatch.setattr("nanovllm.engine.model_runner.load_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(ModelRunner, "warmup_model", lambda self: None)
    monkeypatch.setattr(ModelRunner, "allocate_kv_cache", lambda self: None)
    monkeypatch.setattr(
        ModelRunner,
        "create_attn_backend",
        lambda self, backend_name: DummyAttentionBackend(),
    )

    runner = ModelRunner(config, rank=0, event=[])

    assert isinstance(runner.quant_config, GPTQConfig)
    assert runner.quant_config.quantized_modules == {
        "model.layers.0.self_attn.qkv_proj",
        "model.layers.0.mlp.gate_up_proj",
    }

    layer = runner.model.model.layers[0]
    assert isinstance(layer.self_attn.qkv_proj.quant_method, GPTQLinearMethod)
    assert isinstance(layer.mlp.gate_up_proj.quant_method, GPTQLinearMethod)
    assert layer.self_attn.o_proj.quant_method is None
    assert layer.mlp.down_proj.quant_method is None
