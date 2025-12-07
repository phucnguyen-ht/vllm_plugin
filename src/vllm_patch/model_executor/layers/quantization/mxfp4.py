# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from enum import Enum
from typing import Optional

import torch
from torch.nn.parameter import Parameter

from vllm import envs
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm_patch.model_executor.layers.fused_moe.layer import (
    CustomFusedMoE,
    FusedMoEMethodBase,
)
from vllm_patch.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEQuantConfig, mxfp4_w4a16_moe_quant_config, ocp_mx_moe_quant_config
from vllm_patch.model_executor.layers.fused_moe import modular_kernel as mk

# from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
#     # BatchedMarlinExperts,
#     # MarlinExperts,
#     fused_marlin_moe,
# )
# from vllm_patch.model_executor.layers.fused_moe.gpt_oss_triton_kernels_moe import (
#     OAITritonExperts,
# )
# from vllm.model_executor.layers.fused_moe.trtllm_moe import TrtLlmGenExperts
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
# from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
# from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
#     prepare_moe_fp4_layer_for_marlin,
# )
from vllm_patch.model_executor.layers.quantization.utils.mxfp4_utils import (
    _can_support_mxfp4,
    _swizzle_mxfp4,
)
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

from vllm_patch.utils import (
    has_triton_kernels,
    is_torch_equal_or_newer
)
from vllm.utils import round_up

logger = init_logger(__name__)


# enum for mxfp4 backend
class Mxfp4Backend(Enum):
    NONE = 0

    # FlashInfer Backend
    SM100_FI_MXFP4_MXFP8_TRTLLM = 1
    SM100_FI_MXFP4_MXFP8_CUTLASS = 2
    SM100_FI_MXFP4_BF16 = 3
    SM90_FI_MXFP4_BF16 = 4

    # Marlin Backend
    MARLIN = 5

    # Triton Backend
    TRITON = 6


def get_mxfp4_backend():
    # Backend Selection
    if current_platform.is_cuda():
        raise ValueError("Not supported CUDA platform")
    elif current_platform.is_rocm() and has_triton_kernels():
        logger.info_once("Using Triton backend")
        return Mxfp4Backend.TRITON

    return Mxfp4Backend.NONE

@register_quantization_config("mxfp4")
class Mxfp4Config(QuantizationConfig):
    def __init__(self, ignored_layers: list[str] | None = None):
        super().__init__()
        self.ignored_layers = ignored_layers

    @classmethod
    def from_config(cls, config):
        return cls()

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_name(cls) -> str:
        return "mxfp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        from vllm.attention.layer import Attention  # Avoid circular import

        if isinstance(layer, LinearBase):
            if self.ignored_layers and is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            raise NotImplementedError("Mxfp4 linear layer is not implemented")
        elif isinstance(layer, CustomFusedMoE):
            return Mxfp4MoEMethod(layer.moe_config)
        elif isinstance(layer, Attention):
            raise NotImplementedError("Mxfp4 attention layer is not implemented")
        return None


class Mxfp4MoEMethod(FusedMoEMethodBase):
    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)
        self.topk_indices_dtype = None
        self.moe = moe
        self.mxfp4_backend = get_mxfp4_backend()
        self.max_capture_size = (
            get_current_vllm_config().compilation_config.max_capture_size
        )

        assert self.mxfp4_backend != Mxfp4Backend.NONE, (
            "No MXFP4 MoE backend (FlashInfer/Marlin/Triton) available."
            "Please check your environment and try again."
        )
        self._cache_permute_indices: dict[torch.Size, torch.Tensor] = {}
        print(f"Using Mxfp4MoEMethod with config: {moe = }")

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self.num_experts = num_experts
        weight_dtype = torch.uint8
        scale_dtype = torch.uint8

        # FIXME (zyongye): ship after torch and safetensors support mxfp4
        # is_torch_mxfp4_available = (
        #     hasattr(torch, "float4_e2m1fn_x2") and
        #     hasattr(torch, "float8_e8m0fnu"))
        # if is_torch_mxfp4_available:
        #     weight_dtype = torch.float4_e2m1fn_x2
        #     scale_dtype = torch.float8_e8m0fnu

        mxfp4_block = 32

        intermediate_size_per_partition_after_pad = intermediate_size_per_partition
        if self.mxfp4_backend == Mxfp4Backend.MARLIN:
            raise ValueError("Not supported MOE MARLIN backend")
        elif (
            self.mxfp4_backend == Mxfp4Backend.SM100_FI_MXFP4_MXFP8_TRTLLM
            or self.mxfp4_backend == Mxfp4Backend.SM100_FI_MXFP4_BF16
        ):
            raise ValueError("Not supported MOE TRTLLM backend")
        elif (
            self.mxfp4_backend == Mxfp4Backend.SM100_FI_MXFP4_MXFP8_CUTLASS
            or self.mxfp4_backend == Mxfp4Backend.SM90_FI_MXFP4_BF16
        ):
            raise ValueError("Not supported MOE CUTLASS backend")
        elif current_platform.is_rocm():
            intermediate_size_per_partition_after_pad = round_up(       # 3072
                intermediate_size_per_partition, 256
            )
            hidden_size = round_up(hidden_size, 256)                     # 3072
        else:
            intermediate_size_per_partition_after_pad = round_up(
                intermediate_size_per_partition, 64
            )

        self.intermediate_size = intermediate_size_per_partition_after_pad
        self.hidden_size = hidden_size
        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition_after_pad,
                hidden_size // 2,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition_after_pad,
                hidden_size // mxfp4_block,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w13_bias = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition_after_pad,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_bias", w13_bias)
        set_weight_attrs(w13_bias, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition_after_pad // 2,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition_after_pad // mxfp4_block,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        w2_bias = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_bias", w2_bias)
        set_weight_attrs(w2_bias, extra_weight_attrs)

    def process_weights_after_loading(self, layer):
        if self.mxfp4_backend == Mxfp4Backend.MARLIN:
            raise ValueError("Not supported MOE Marlin backend")
        elif (
            self.mxfp4_backend == Mxfp4Backend.SM100_FI_MXFP4_MXFP8_TRTLLM
            or self.mxfp4_backend == Mxfp4Backend.SM100_FI_MXFP4_BF16
        ):
            raise ValueError("Not supported MOE FlashInfer backend")
        elif (
            self.mxfp4_backend == Mxfp4Backend.SM100_FI_MXFP4_MXFP8_CUTLASS
            or self.mxfp4_backend == Mxfp4Backend.SM90_FI_MXFP4_BF16
        ):
            raise ValueError("Not supported MOE CUTLASS backend")
            
        elif self.mxfp4_backend == Mxfp4Backend.TRITON:
            from triton_kernels.matmul_ogs import FlexCtx, PrecisionConfig

            w13_bias = layer.w13_bias.to(torch.float32)
            w2_bias = layer.w2_bias.to(torch.float32)

            layer.w13_bias = Parameter(w13_bias, requires_grad=False)
            layer.w2_bias = Parameter(w2_bias, requires_grad=False)

            # Ideally we'd use FusedMoEModularKernel.prepare_finalize object
            # (stored in self.fused_experts) to determine if the MoE has a
            # batched activation format. As self.fused_experts is not
            # initialized at this point, we resort to checking the MoE config
            # directly.
            is_batched_moe = self.moe.use_pplx_kernels or self.moe.use_deepep_ll_kernels
            if is_batched_moe:
                num_warps = 4 if envs.VLLM_MOE_DP_CHUNK_SIZE <= 512 else 8
            else:
                num_warps = 8
                
            print(f"{num_warps = }, {self.moe.use_pplx_kernels = }, {self.moe.use_deepep_ll_kernels = }")

            w13_weight, w13_flex, w13_scale = _swizzle_mxfp4(
                layer.w13_weight, layer.w13_weight_scale, num_warps
            )
            w2_weight, w2_flex, w2_scale = _swizzle_mxfp4(
                layer.w2_weight, layer.w2_weight_scale, num_warps
            )

            self.w13_precision_config = PrecisionConfig(
                weight_scale=w13_scale, flex_ctx=FlexCtx(rhs_data=w13_flex)
            )
            self.w2_precision_config = PrecisionConfig(
                weight_scale=w2_scale, flex_ctx=FlexCtx(rhs_data=w2_flex)
            )

            self.w13_weight_triton_tensor = w13_weight
            self.w2_weight_triton_tensor = w2_weight
            
            # ADD
            self.moe_quant_config = mxfp4_w4a16_moe_quant_config(
                w1_bias=layer.w13_bias,
                w2_bias=layer.w2_bias,
                w1_scale=self.w13_precision_config,
                w2_scale=self.w2_precision_config,
            )

            # need to delete the original weights to save memory on single GPU
            del layer.w13_weight
            del layer.w2_weight
            layer.w13_weight = None
            layer.w2_weight = None
            torch.cuda.empty_cache()
        else:
            raise ValueError(f"Unsupported backend: {self.mxfp4_backend}")

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        if self.mxfp4_backend == Mxfp4Backend.MARLIN:
            return mxfp4_w4a16_moe_quant_config(
                w1_bias=layer.w13_bias,
                w2_bias=layer.w2_bias,
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
            )
        elif self.mxfp4_backend == Mxfp4Backend.TRITON:
            w1_scale = self.w13_precision_config
            w2_scale = self.w2_precision_config
            return mxfp4_w4a16_moe_quant_config(
                w1_bias=layer.w13_bias,
                w2_bias=layer.w2_bias,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
            )
        else:
            w1_scale = layer.w13_weight_scale
            w2_scale = layer.w2_weight_scale
            return ocp_mx_moe_quant_config(
                quant_dtype="mxfp4",
                w1_bias=layer.w13_bias,
                w2_bias=layer.w2_bias,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
            )

    # def select_gemm_impl(
    #     self,
    #     prepare_finalize: mk.FusedMoEPrepareAndFinalize,
    #     layer: torch.nn.Module,
    # ) -> mk.FusedMoEPermuteExpertsUnpermute:
    #     if (
    #         prepare_finalize.activation_format
    #         == mk.FusedMoEActivationFormat.BatchedExperts
    #     ):
    #         if self.mxfp4_backend == Mxfp4Backend.MARLIN:
    #             max_num_tokens_per_rank = prepare_finalize.max_num_tokens_per_rank()
    #             assert max_num_tokens_per_rank is not None
    #             assert self.moe_quant_config is not None
    #             return BatchedMarlinExperts(
    #                 max_num_tokens=max_num_tokens_per_rank,
    #                 num_dispatchers=prepare_finalize.num_dispatchers(),
    #                 quant_config=self.moe_quant_config,
    #             )
    #         else:
    #             raise NotImplementedError(
    #                 "Incompatible Mxfp4 backend for EP batched experts format"
    #             )
    #     else:
    #         assert self.moe_quant_config is not None
    #         if (
    #             self.mxfp4_backend == Mxfp4Backend.SM100_FI_MXFP4_MXFP8_TRTLLM
    #             or self.mxfp4_backend == Mxfp4Backend.SM100_FI_MXFP4_BF16
    #         ):
    #             # B200 code-path
    #             kwargs = {
    #                 "gemm1_alpha": layer.gemm1_alpha,
    #                 "gemm1_beta": layer.gemm1_beta,
    #                 "gemm1_clamp_limit": layer.gemm1_clamp_limit,
    #                 # TODO(bnell): part of quant_config
    #                 "max_capture_size": self.max_capture_size,
    #             }
    #             return TrtLlmGenExperts(self.moe, self.moe_quant_config, **kwargs)
    #         elif self.mxfp4_backend == Mxfp4Backend.MARLIN:
    #             return MarlinExperts(self.moe_quant_config)
    #         else:
    #             return OAITritonExperts(self.moe_quant_config)

    def _route_and_experts(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: torch.Tensor | None = None,
        logical_to_physical_map: torch.Tensor | None = None,
        logical_replica_count: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert isinstance(self.fused_experts, mk.FusedMoEModularKernel)

        topk_weights, topk_ids, _ = CustomFusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
            indices_type=self.topk_indices_dtype,
            enable_eplb=enable_eplb,
            expert_map=expert_map,
            expert_load_view=expert_load_view,
            logical_to_physical_map=logical_to_physical_map,
            logical_replica_count=logical_replica_count,
        )

        w13_weight = (
            self.w13_weight_triton_tensor
            if layer.w13_weight is None
            else layer.w13_weight
        )
        w2_weight = (
            self.w2_weight_triton_tensor if layer.w2_weight is None else layer.w2_weight
        )
        assert all([w is not None for w in [w13_weight, w2_weight]])

        return self.fused_experts(
            hidden_states=x,
            w1=w13_weight,
            w2=w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            activation=activation,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: torch.Tensor | None = None,
        logical_to_physical_map: torch.Tensor | None = None,
        logical_replica_count: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if enable_eplb:
            raise NotImplementedError("EPLB is not supported for mxfp4")

        if self.fused_experts is not None:
            return self._route_and_experts(
                layer,
                x,
                router_logits,
                top_k,
                renormalize,
                use_grouped_topk,
                topk_group,
                num_expert_group,
                global_num_experts,
                expert_map,
                custom_routing_function,
                scoring_func,
                e_score_correction_bias,
                apply_router_weight_on_input,
                activation,
                enable_eplb,
                expert_load_view,
                logical_to_physical_map,
                logical_replica_count,
            )

        if self.mxfp4_backend == Mxfp4Backend.MARLIN:
            raise ValueError("Not supported MOE Marlin backend")

        assert _can_support_mxfp4(
            use_grouped_topk,
            topk_group,
            num_expert_group,
            expert_map,
            custom_routing_function,
            e_score_correction_bias,
            apply_router_weight_on_input,
            scoring_func,
            activation,
            expert_load_view,
            logical_to_physical_map,
            logical_replica_count,
        ), "MXFP4 are not supported with this configuration."

        if (
            self.mxfp4_backend == Mxfp4Backend.SM100_FI_MXFP4_MXFP8_TRTLLM
            or self.mxfp4_backend == Mxfp4Backend.SM100_FI_MXFP4_BF16
        ):
            raise ValueError("Not supported TRTLLM backend")
        elif (
            self.mxfp4_backend == Mxfp4Backend.SM100_FI_MXFP4_MXFP8_CUTLASS
            or self.mxfp4_backend == Mxfp4Backend.SM90_FI_MXFP4_BF16
        ):
            raise ValueError("Not supported CUTLASS backend")

        elif self.mxfp4_backend == Mxfp4Backend.TRITON:
            from vllm_patch.model_executor.layers.fused_moe.gpt_oss_triton_kernels_moe import (  # noqa: E501
                triton_kernel_moe_forward,
            )
            
            # print("Using triton forward...")
            return triton_kernel_moe_forward(
                hidden_states=x,
                w1=self.w13_weight_triton_tensor,
                w2=self.w2_weight_triton_tensor,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
                quant_config=self.moe_quant_config,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )
        else:
            raise ValueError(f"Unsupported backend: {self.mxfp4_backend}")
