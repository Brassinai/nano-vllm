"""Model-weight quantization interfaces and registry."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Callable

import torch


class QuantizeMethodBase(ABC):
    """Weights and compute implementation for one quantized layer type."""

    @abstractmethod
    def create_weights(self, layer: torch.nn.Module) -> None:
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        return


class QuantizationConfig(ABC):
    """Backend config parsed from the model's quantization metadata."""

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_config_filenames(cls) -> tuple[str, ...]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_config(cls, raw_config: dict[str, Any]) -> "QuantizationConfig":
        raise NotImplementedError

    @classmethod
    def get_from_keys(cls, config: dict[str, Any], keys: tuple[str, ...]) -> Any:
        for key in keys:
            if key in config:
                return config[key]
        raise ValueError(f"Missing one of {keys!r} in quantization config.")

    @classmethod
    def get_from_keys_or(
        cls,
        config: dict[str, Any],
        keys: tuple[str, ...],
        default: Any,
    ) -> Any:
        try:
            return cls.get_from_keys(config, keys)
        except ValueError:
            return default

    @abstractmethod
    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> QuantizeMethodBase | None:
        raise NotImplementedError

    def validate_runtime(self, dtype: torch.dtype) -> None:
        return

    def update_from_model_path(self, model_path: str) -> None:
        return


class QuantizationRegistry:
    """Registry for model-weight quantization backends."""

    _registry: dict[str, type[QuantizationConfig]] = {}

    @classmethod
    def register(
        cls,
        name: str,
    ) -> Callable[[type[QuantizationConfig]], type[QuantizationConfig]]:
        def decorator(config_cls: type[QuantizationConfig]) -> type[QuantizationConfig]:
            if not issubclass(config_cls, QuantizationConfig):
                raise TypeError("Quantization configs must extend QuantizationConfig.")
            if name in cls._registry:
                raise ValueError(
                    f"Model quantization backend {name!r} is already registered."
                )
            cls._registry[name] = config_cls
            return config_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[QuantizationConfig]:
        try:
            return cls._registry[name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown model quantization backend {name!r}. "
                f"Available backends: {sorted(cls._registry)}"
            ) from exc

    @classmethod
    def list_backends(cls) -> list[str]:
        return sorted(cls._registry)


def _load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"Quantization config at {path!r} must be a JSON object.")
    return value


def _find_raw_config(
    model_path: str,
    hf_config: Any,
    config_cls: type[QuantizationConfig],
) -> dict[str, Any]:
    raw_hf_config = getattr(hf_config, "quantization_config", None)
    if isinstance(raw_hf_config, dict):
        return raw_hf_config

    for filename in config_cls.get_config_filenames():
        path = os.path.join(model_path, filename)
        if os.path.isfile(path):
            return _load_json(path)

    raise FileNotFoundError(
        f"Could not find quantization metadata for {config_cls.get_name()!r}. "
        f"Expected hf_config.quantization_config or one of "
        f"{config_cls.get_config_filenames()} under {model_path!r}."
    )


def resolve_quantization_config(
    backend_name: str | None,
    model_path: str,
    hf_config: Any,
) -> QuantizationConfig | None:
    """Resolve an explicit model quantization backend from local metadata."""

    if backend_name is None or backend_name == "none":
        return None
    config_cls = QuantizationRegistry.get(backend_name)
    config = config_cls.from_config(_find_raw_config(model_path, hf_config, config_cls))
    config.update_from_model_path(model_path)
    return config
