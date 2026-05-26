import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.quantization.base import QuantizationConfig, QuantizeMethodBase


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


def _copy_exact(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
    if param.shape != loaded_weight.shape:
        raise ValueError(
            f"Loaded tensor shape {tuple(loaded_weight.shape)} does not match "
            f"parameter shape {tuple(param.shape)}."
        )
    param.data.copy_(loaded_weight)


class LinearBase(nn.Module):
    supports_weight_quantization = True

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        full_input_size: int | None = None,
        full_output_size: int | None = None,
        input_partition_start: int = 0,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.input_size = input_size
        self.output_size = output_size
        self.full_input_size = full_input_size or input_size
        self.full_output_size = full_output_size or output_size
        self.input_partition_start = input_partition_start
        self.prefix = prefix
        self.quant_config = quant_config
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

        self.quant_method: QuantizeMethodBase | None = None
        if quant_config is not None:
            self.quant_method = quant_config.get_quant_method(self, prefix)
        if self.quant_method is None:
            self.weight = nn.Parameter(torch.empty(output_size, input_size))
            self.weight.weight_loader = self.weight_loader
        else:
            self.quant_method.create_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__(
            input_size,
            output_size,
            bias,
            quant_config=quant_config,
            prefix=prefix,
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        _copy_exact(param, loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quant_method is not None:
            return self.quant_method.apply(self, x, self.bias)
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        tp_size = dist.get_world_size()
        super().__init__(
            input_size,
            divide(output_size, tp_size),
            bias,
            0,
            quant_config=quant_config,
            prefix=prefix,
            full_input_size=input_size,
            full_output_size=output_size,
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quant_method is not None:
            return self.quant_method.apply(self, x, self.bias)
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        self.output_sizes = output_sizes
        super().__init__(
            input_size,
            sum(output_sizes),
            bias,
            quant_config=quant_config,
            prefix=prefix,
        )

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int,
    ):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)

class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        self.output_sizes = [
            total_num_heads * self.head_size,
            total_num_kv_heads * self.head_size,
            total_num_kv_heads * self.head_size,
        ]
        output_size = sum(self.output_sizes)
        super().__init__(
            hidden_size,
            output_size,
            bias,
            quant_config=quant_config,
            prefix=prefix,
        )

    def _qkv_shard(self, shard_id: str) -> tuple[int, int]:
        assert shard_id in ["q", "k", "v"]
        if shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = (
                self.num_heads * self.head_size
                + self.num_kv_heads * self.head_size
            )
        return shard_offset, shard_size

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str,
    ):
        shard_offset, shard_size = self._qkv_shard(loaded_shard_id)
        param_data = param.data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)

class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        tp_size = dist.get_world_size()
        local_input_size = divide(input_size, tp_size)
        super().__init__(
            local_input_size,
            output_size,
            bias,
            1,
            quant_config=quant_config,
            prefix=prefix,
            full_input_size=input_size,
            full_output_size=output_size,
            input_partition_start=dist.get_rank() * local_input_size,
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quant_method is not None:
            y = self.quant_method.apply(
                self,
                x,
                self.bias if self.tp_rank == 0 else None,
            )
        else:
            y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
