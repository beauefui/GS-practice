"""
激活值提取 Hooks

通过 PyTorch 的 register_forward_hook 机制, 在模型前向传播时
自动捕获指定层的激活值 (无需修改模型源码)。

Gemma 模型层级结构 (以 Gemma 3 1B 为例):
  model
  ├── embed_tokens                          # 词嵌入层
  ├── layers                                # 26 层 GemmaDecoderLayer
  │   ├── [0] GemmaDecoderLayer
  │   │   ├── input_layernorm               # RMSNorm (attention 前)
  │   │   ├── self_attn                     # 自注意力
  │   │   │   ├── q_proj, k_proj, v_proj    # QKV 投影
  │   │   │   └── o_proj                    # 输出投影
  │   │   ├── post_attention_layernorm      # RMSNorm (MLP 前)
  │   │   └── mlp                           # 前馈网络
  │   │       ├── gate_proj, up_proj        # 门控投影
  │   │       └── down_proj                 # 下投影
  │   ├── [1] ...
  │   └── [25] ...
  └── norm                                  # 最终 RMSNorm

Hook 点位说明:
  - "residual": 整个 decoder layer 的输出 (residual stream)
                → hook 注册在 model.layers[i]
  - "mlp_output": MLP 子层的输出
                  → hook 注册在 model.layers[i].mlp
  - "attn_output": 自注意力子层的输出
                   → hook 注册在 model.layers[i].self_attn
"""

import torch
import torch.nn as nn
from typing import Literal


# 支持的 hook 点位类型
HookPoint = Literal["residual", "mlp_output", "attn_output"]


def _get_target_module(
    model: nn.Module,
    layer_idx: int,
    hook_point: HookPoint,
) -> nn.Module:
    """
    根据层索引和 hook 点位, 获取要注册 hook 的目标模块

    Args:
        model: Gemma 模型 (AutoModelForCausalLM 加载的)
        layer_idx: 要 hook 的层索引 (0-indexed)
        hook_point: hook 点位类型

    Returns:
        目标 nn.Module 对象
    """
    # 获取模型的 decoder layers
    # transformers 的 Gemma 模型: model.model.layers[i]
    layers = model.model.layers

    if layer_idx < 0 or layer_idx >= len(layers):
        raise ValueError(
            f"layer_idx={layer_idx} 超出范围, 模型共有 {len(layers)} 层 (0-{len(layers)-1})"
        )

    layer = layers[layer_idx]

    if hook_point == "residual":
        # hook 整个 decoder layer → 捕获 residual stream
        return layer
    elif hook_point == "mlp_output":
        # hook MLP 子层
        return layer.mlp
    elif hook_point == "attn_output":
        # hook 自注意力子层
        return layer.self_attn
    else:
        raise ValueError(f"未知的 hook_point: {hook_point}, 支持: residual, mlp_output, attn_output")


class ActivationCache:
    """
    激活值缓存 — 通过 context manager 自动管理 hooks 的生命周期

    用法:
        with ActivationCache(model, layer_idx=20, hook_point="residual") as cache:
            outputs = model(**inputs)
            activations = cache.get()  # shape: (batch, seq_len, d_model)

    工作原理:
      1. __enter__: 在目标模块上注册 forward_hook
      2. 模型前向传播时, hook 自动捕获该模块的输出
      3. __exit__: 自动移除 hook, 不影响模型
    """

    def __init__(
        self,
        model: nn.Module,
        layer_idx: int,
        hook_point: HookPoint = "residual",
    ):
        """
        Args:
            model: Gemma 模型
            layer_idx: 要 hook 的层索引
            hook_point: hook 点位 ("residual" / "mlp_output" / "attn_output")
        """
        self.model = model
        self.layer_idx = layer_idx
        self.hook_point = hook_point

        # 内部状态
        self._activation: torch.Tensor | None = None
        self._hook_handle = None

    def _hook_fn(self, module: nn.Module, input: tuple, output) -> None:
        """
        Hook 回调函数 — 在目标模块前向传播完成后被调用

        Args:
            module: 被 hook 的模块
            input: 模块的输入 (tuple)
            output: 模块的输出
                    - 对于 decoder layer: output 是 tuple, output[0] 是 hidden states
                    - 对于 mlp: output 是 tensor
                    - 对于 self_attn: output 是 tuple, output[0] 是 attn output
        """
        # decoder layer 和 self_attn 的输出是 tuple, 取第一个元素 (hidden states)
        if isinstance(output, tuple):
            self._activation = output[0].detach()
        else:
            self._activation = output.detach()

    def __enter__(self) -> "ActivationCache":
        """注册 hook"""
        target_module = _get_target_module(self.model, self.layer_idx, self.hook_point)
        self._hook_handle = target_module.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """移除 hook"""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def get(self) -> torch.Tensor:
        """
        获取捕获到的激活值

        Returns:
            activation: shape (batch_size, seq_len, d_model)

        Raises:
            RuntimeError: 如果还没有执行前向传播
        """
        if self._activation is None:
            raise RuntimeError(
                "还没有捕获到激活值! 请先在 with 块内执行 model(**inputs)"
            )
        return self._activation

    def clear(self) -> None:
        """清除缓存的激活值 (释放显存)"""
        self._activation = None


class MultiLayerActivationCache:
    """
    多层激活值缓存 — 同时 hook 多个层

    用法:
        layers = [7, 13, 17, 22]
        with MultiLayerActivationCache(model, layers, hook_point="residual") as cache:
            outputs = model(**inputs)
            acts = cache.get()  # dict: {layer_idx: tensor}

        # 也可以获取堆叠的张量
        with MultiLayerActivationCache(model, layers) as cache:
            outputs = model(**inputs)
            stacked = cache.get_stacked()  # shape: (num_layers, batch, seq_len, d_model)
    """

    def __init__(
        self,
        model: nn.Module,
        layer_indices: list[int],
        hook_point: HookPoint = "residual",
    ):
        self.caches = [
            ActivationCache(model, layer_idx=idx, hook_point=hook_point)
            for idx in layer_indices
        ]
        self.layer_indices = layer_indices

    def __enter__(self) -> "MultiLayerActivationCache":
        for cache in self.caches:
            cache.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for cache in self.caches:
            cache.__exit__(exc_type, exc_val, exc_tb)

    def get(self) -> dict[int, torch.Tensor]:
        """获取各层的激活值, 返回 {layer_idx: tensor} 字典"""
        return {idx: cache.get() for idx, cache in zip(self.layer_indices, self.caches)}

    def get_stacked(self) -> torch.Tensor:
        """获取堆叠的激活值张量, shape: (num_layers, batch, seq_len, d_model)"""
        return torch.stack([cache.get() for cache in self.caches], dim=0)

    def clear(self) -> None:
        """清除所有缓存"""
        for cache in self.caches:
            cache.clear()
