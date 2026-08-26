# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

from .base import MxFp4LinearKernel, MxFp4LinearLayerConfig


class AiterMxFp4LinearKernel(MxFp4LinearKernel):
    """MXFP4 (W4A4) dense linear via aiter's CK a4w4 GEMM on ROCm.

    Weights (E2M1) and their per-group-32 E8M0 scales are preshuffled into the
    AITER CK layout at load time; activations are dynamically quantized to FP4
    with per-1x32 E8M0 scales at apply time. This is the dense/shared-expert
    counterpart of the aiter fused-MoE MXFP4 path.
    """

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        from vllm._aiter_ops import rocm_aiter_ops

        if not current_platform.is_rocm():
            return False, "not ROCm"
        if not rocm_aiter_ops.is_enabled():
            return False, "aiter disabled"
        # The CK a4w4 kernel is not built for gfx942.
        from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx

        if get_gfx() == "gfx942":
            return False, "a4w4 not supported on gfx942"
        return True, None

    @classmethod
    def can_implement(cls, c: MxFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from aiter.ops.shuffle import shuffle_scale, shuffle_weight

        # weight: packed uint8 [N, K/2] (2 FP4 per byte); scale: E8M0 [N, K/32].
        w = shuffle_weight(
            layer.weight.data.view(torch.float4_e2m1fn_x2), layout=(16, 16)
        )
        s = shuffle_scale(layer.weight_scale.data.view(torch.uint8))
        replace_parameter(layer, "weight", w)
        replace_parameter(layer, "weight_scale", s)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        import aiter

        quant = aiter.get_triton_quant(aiter.QuantType.per_1x32)
        n = layer.output_size_per_partition
        out_shape = (*x.shape[:-1], n)
        x_2d = x.reshape(-1, x.shape[-1])
        x_q, x_scale = quant(x_2d, shuffle=True)
        y = aiter.gemm_a4w4(
            x_q,
            layer.weight,
            x_scale,
            layer.weight_scale,
            dtype=x.dtype,
            bpreshuffle=True,
        )[: x_2d.shape[0]]
        if bias is not None:
            y = y + bias
        return y.reshape(out_shape)
