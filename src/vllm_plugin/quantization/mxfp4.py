# TODO(anhduong): Refactor to import only what is needed instead of wildcard import
from vllm_patch.model_executor.layers.quantization.mxfp4 import *
from vllm_plugin.fused_moe.layer import MorehFusedMoE
from vllm.model_executor.layers.quantization import (register_quantization_config)
from vllm_patch.model_executor.layers.quantization.utils.mxfp4_utils import (
    _can_support_mxfp4, _swizzle_mxfp4)

from vllm.utils import round_up

from vllm.logger import init_logger
from vllm_patch.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm_plugin import envs as moreh_envs
from vllm_plugin.fused_moe.rocm_moreh_fused_moe import rocm_gptoss_moreh_moe_1stage
# from vllm_moreh.quantization.utils.fused_moe_tuning import (
#     get_best_tuning_config,
#     is_fused_moe_1stage_better_than_2stages,
# )

from aiter import get_hip_quant, dtypes, QuantType

SHUFFLE_IN = 16
SHUFFLE_IK = 16

logger = init_logger(__name__)

def is_moreh_dual_moe_enabled():
    return moreh_envs.VLLM_MOREH_USE_DUAL_MOE

@register_quantization_config("moreh-mxfp4")
class MorehMxfp4Config(Mxfp4Config):
    @classmethod
    def get_name(cls) -> str:
        return "moreh-mxfp4"

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        from vllm.attention.layer import Attention  # Avoid circular import

        if isinstance(layer, LinearBase):
            if self.ignored_layers and is_layer_skipped(
                    prefix=prefix,
                    ignored_layers=self.ignored_layers,
                    fused_mapping=self.packed_modules_mapping):
                return UnquantizedLinearMethod()
            raise NotImplementedError("Mxfp4 linear layer is not implemented")
        elif isinstance(layer, MorehFusedMoE):
            return MorehMxfp4MoEMethod(layer.moe_config)
        elif isinstance(layer, Attention):
            raise NotImplementedError(
                "Mxfp4 attention layer is not implemented")
        return None


class MorehMxfp4MoEMethod(Mxfp4MoEMethod):
    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # print extra_weight_attrs keys and values for debugging
        # print(f"MorehMxfp4MoEMethod.create_weights extra_weight_attrs: {extra_weight_attrs}")
        # self.intermediate_size_full = extra_weight_attrs.pop("intermediate_size_full")
        # self.intermediate_size_per_partition_before_pad = extra_weight_attrs.pop("intermediate_size_per_partition_before_pad")
        self.tp_size = extra_weight_attrs.pop("tp_size")
        
        intermediate_size_per_partition_after_pad = round_up(       # 3072
            intermediate_size_per_partition, 256
        )
        hidden_size = round_up(hidden_size, 256)                     # 3072
        
        self.num_experts = num_experts
        weight_dtype = torch.uint8
        scale_dtype = torch.uint8
        mxfp4_block = 32
        
        logger.info_once(f"{hidden_size = }, {num_experts = }, {intermediate_size_per_partition = }, {extra_weight_attrs = }")
        # Not roundup to pad
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
        
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
             
        if enable_eplb:
            raise NotImplementedError("EPLB is not supported for mxfp4")

        assert self.fused_experts is None, "self.fused_experts should be None"

        assert _can_support_mxfp4(
            use_grouped_topk, topk_group, num_expert_group, expert_map,
            custom_routing_function, e_score_correction_bias,
            apply_router_weight_on_input, scoring_func, activation,
            expert_load_view, logical_to_physical_map,
            logical_replica_count), (
                "MXFP4 are not supported with this configuration.")

        if is_moreh_dual_moe_enabled():
            from aiter.fused_moe import fused_topk
            
            topk_weights, topk_ids = fused_topk(x, router_logits, top_k, False)
            group_size = 32
            
            logger.info_once(f"[Moreh MXFP4 debug] {x.shape = }, {router_logits.shape = }, {top_k = }, {topk_ids.shape = }, {topk_weights.shape = }, {torch.topk(router_logits, 5) = }")
            logger.info_once(f"[Moreh MXFP4 debug] {x.dtype = }, {self.w1_qweight_shuffled.dtype = }, {self.w2_qweight_shuffled = }"
                             f"{layer.w13_weight_scale.dtype = }, {layer.w2_weight_scale.dtype = }, {layer.w13_bias.dtype = }, {layer.w2_bias.dtype = }")
            
            kernel_kwargs = dict(
                hidden_states=x,
                w1=self.w1_qweight_shuffled,
                w2=self.w2_qweight_shuffled,
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                w1_bias=layer.w13_bias,
                w2_bias=layer.w2_bias,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
                tuning_config=None,
                block_shape=[0, group_size],
                a_scale=None
            )

            return rocm_gptoss_moreh_moe_1stage(**kernel_kwargs)
            
        else:
            # mxfp4 path
            # routing + 2-staged MoE with triton kernels
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

    def process_weights_after_loading(self, layer):
        if is_moreh_dual_moe_enabled():
            # from triton_kernels.matmul_ogs import FlexCtx, PrecisionConfig
            
            

            # w13_bias = layer.w13_bias.to(torch.float32)
            # w2_bias = layer.w2_bias.to(torch.float32)

            # layer.w13_bias = Parameter(w13_bias, requires_grad=False)
            # layer.w2_bias = Parameter(w2_bias, requires_grad=False)

            # # FIXME warp need to be adjusted based on batch size
            # # only apply to  batched mode
            # if self.moe.use_ep:
            #     num_warps = 4 if envs.VLLM_MOE_DP_CHUNK_SIZE <= 512 else 8
            # else:
            #     num_warps = 8

            # def clone_and_upcast_mxfp_weight(weight, scale, axis=1, target_dtype=torch.bfloat16, target_device="cuda:0"):
            #     """
            #     Clone MXFP4 weight, move to device cuda:0, upcast to target dtype, then move back to target device.
            #     """
            #     # Avoid in-place operation on the original tensor
            #     # upcast_from_mxfp only works on cuda:0
            #     weight_clone = weight.data.clone().to("cuda:0")
            #     scale_clone = scale.data.clone().to("cuda:0")

            #     from triton_kernels.numerics_details.mxfp import upcast_from_mxfp
            #     upcasted = upcast_from_mxfp(weight_clone, scale_clone, target_dtype, axis=axis)

            #     # Cleanup clones immediately after use
            #     del weight_clone, scale_clone
            #     return Parameter(upcasted.to(target_device)) # move back to target device

            # target_device = layer.w13_weight.data.device
            # target_dtype = torch.bfloat16

            # # NOTE(anhtt): Upcast weights to bf16 for dual MoE
            # w13_bf16_weight = clone_and_upcast_mxfp_weight(
            #     layer.w13_weight,
            #     layer.w13_weight_scale,
            #     axis=2,
            #     target_dtype=target_dtype,
            #     target_device=target_device,
            # )
            # w2_bf16_weight = clone_and_upcast_mxfp_weight(
            #     layer.w2_weight,
            #     layer.w2_weight_scale,
            #     axis=2,
            #     target_dtype=target_dtype,
            #     target_device=target_device,
            # )
            
            # # Clone bias as BF16
            # w13_fp8_bias = layer.w13_bias.data.clone().to(torch.bfloat16)
            # w2_fp8_bias = layer.w2_bias.data.clone().to(torch.bfloat16)
            
            # w1_qweight, w1_scales = get_hip_quant(QuantType.per_Token)(w13_bf16_weight, quant_dtype=dtypes.fp8)
            # w2_qweight, w2_scales = get_hip_quant(QuantType.per_Token)(w2_bf16_weight, quant_dtype=dtypes.fp8)

            # # No need padding anymore because vllm already default pad to 256
            
            # # Finally assign to self
            # from aiter.ops.shuffle import shuffle_weight
            
            # self.w13_fp8_weight = shuffle_weight(w1_qweight, (16, 16))
            # self.w13_fp8_weight_scale = w1_scales
            # self.w13_fp8_bias = w13_fp8_bias
            
            # self.w2_fp8_weight = shuffle_weight(w2_qweight, (16, 16))
            # self.w2_fp8_weight_scale = w2_scales
            # self.w2_fp8_bias = w2_fp8_bias
            
            # # Cleanup
            # torch.cuda.empty_cache()

            # # NOTE(anhtt): uint8 -> MXFP4
            # w13_weight, w13_flex, w13_scale = _swizzle_mxfp4(
            #     layer.w13_weight, layer.w13_weight_scale, num_warps)
            # w2_weight, w2_flex, w2_scale = _swizzle_mxfp4(
            #     layer.w2_weight, layer.w2_weight_scale, num_warps)

            # self.w13_precision_config = PrecisionConfig(
            #     weight_scale=w13_scale, flex_ctx=FlexCtx(rhs_data=w13_flex))
            # self.w2_precision_config = PrecisionConfig(
            #     weight_scale=w2_scale, flex_ctx=FlexCtx(rhs_data=w2_flex))

            # # Only keep Triton weight tensors when they are actually needed:
            # # - non-dual path (fallback to Triton / modular kernels), or
            # # - dual path with tp_size > 1 (large-batch fallback uses Triton).
            # if (not is_moreh_dual_moe_enabled()) or getattr(self, "tp_size", 1) > 1:
            #     self.w13_weight_triton_tensor = w13_weight
            #     self.w2_weight_triton_tensor = w2_weight
            # else:
            #     # TP=1 dual MoE never falls back to Triton; free these tensors.
            #     del w13_weight, w2_weight

            # # need to delete the original weights to save memory on single GPU
            # del layer.w13_weight
            # del layer.w2_weight
            # layer.w13_weight = None
            # layer.w2_weight = None
            # torch.cuda.empty_cache()

            # Remove negative zero from w1_qweight
            w1_qweight = layer.w13_weight       # (e, 2 * n, k / 2)
            w2_qweight = layer.w2_weight        # (e, k, n / 2)
            # w1_scales = layer.w13_weight_scale  # (e, 2 * n, k / 2)
            # w2_scales = layer.w2_weight_scale   # (e, k, n / 32)
            
            e, n = layer.w13_bias.shape; n = n // 2
            _, k = layer.w2_bias.shape
            
            w1_qweight_left = w1_qweight & 0b11110000
            w1_qweight_right = w1_qweight & 0b00001111
            w1_qweight_left[w1_qweight_left == 0b10000000] = 0
            w1_qweight_right[w1_qweight_right == 0b00001000] = 0
            w1_qweight = w1_qweight_left | w1_qweight_right

            # Remove negative zero from w2_qweight
            w2_qweight_left = w2_qweight & 0b11110000
            w2_qweight_right = w2_qweight & 0b00001111
            w2_qweight_left[w2_qweight_left == 0b10000000] = 0
            w2_qweight_right[w2_qweight_right == 0b00001000] = 0
            w2_qweight = w2_qweight_left | w2_qweight_right

            K_UNROLL = 4
            MFMA_SIZE_MN = 16
            MXFP4_BLOCK_SIZE = 32
            
            # print(f"{e = }, {n = }, {k = }, {type(w1_qweight) = }, {w1_qweight.shape = }, {w1_qweight.dtype = }, {w1_qweight.device = }")
            # print(f"{e = }, {n = }, {k = }, {type(w2_qweight) = }, {w2_qweight.shape = }, {w2_qweight.dtype = }, {w2_qweight.device = }")
            # shuffle w1_qweight
            self.w1_qweight_shuffled = w1_qweight.view(e, 2 * n, (k // 2) // 64, K_UNROLL, 4, 4).transpose(-3, -2).reshape(e, 2 * n, (k // 2))
            # shuffle w2_qweight
            self.w2_qweight_shuffled = w2_qweight.view(e, k, (n // 2) // 64, K_UNROLL, 4, 4).transpose(-3, -2).reshape(e, k, n // 2)
            
            del layer.w13_weight
            del layer.w2_weight
            layer.w13_weight = None
            layer.w2_weight = None
            torch.cuda.empty_cache()
        else:
            super().process_weights_after_loading(layer)
            
    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        return mxfp4_w4a16_moe_quant_config(
            w1_bias=layer.w13_bias,
            w2_bias=layer.w2_bias,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
        )