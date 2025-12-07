from vllm_patch.model_executor.layers.fused_moe.layer import *
from vllm.utils import round_up
from vllm.config import VllmConfig

from vllm_patch.model_executor.layers.fused_moe.rocm_aiter_fused_moe import is_rocm_aiter_fusion_shared_expert_enabled
from vllm_patch.model_executor.layers.fused_moe.config import FusedMoEParallelConfig, FusedMoEQuantConfig

from vllm_plugin.fused_moe.config import MorehFusedMoEParallelConfig, MorehFusedMoEConfig

# from vllm_plugin.utils import has_mori
from vllm_plugin import envs as moreh_envs

class MorehFusedMoE(CustomFusedMoE):
    def __init__(
        self,
        num_experts: int,  # Global number of experts
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype | None = None,
        reduce_results: bool = False,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: int | None = None,
        topk_group: int | None = None,
        quant_config: QuantizationConfig | None = None,
        tp_size: int | None = None,
        ep_size: int | None = None,
        dp_size: int | None = None,
        prefix: str = "",
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        num_redundant_experts: int = 0,
        has_bias: bool = False,
        is_sequence_parallel=False,
        zero_expert_num: int | None = 0,
        zero_expert_type: str | None = None,
        expert_mapping: list[tuple[str, str, int, str]] | None = None,
        n_shared_experts: int | None = None,
    ):
        CustomOp.__init__(self)
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        vllm_config = get_current_vllm_config()

        # FIXME (varun): We should have a better way of inferring the activation
        # datatype. This works for now as the tensor datatype entering the MoE
        # operation is typically unquantized (i.e. float16/bfloat16).
        if vllm_config.model_config is not None:
            moe_in_dtype = vllm_config.model_config.dtype
        else:
            # TODO (bnell): This is a hack to get test_mixtral_moe to work
            # since model_config is not set in the pytest test.
            moe_in_dtype = params_dtype

        tp_size_ = (
            tp_size if tp_size is not None else get_tensor_model_parallel_world_size()
        )
        dp_size_ = dp_size if dp_size is not None else get_dp_group().world_size

        self.is_sequence_parallel = is_sequence_parallel
        self.sp_size = tp_size_ if is_sequence_parallel else 1

        self.moe_parallel_config: MorehFusedMoEParallelConfig = MorehFusedMoEParallelConfig.make(
            tp_size_=tp_size_,
            dp_size_=dp_size_,
            vllm_parallel_config=vllm_config.parallel_config,
        )

        self.global_num_experts = num_experts + num_redundant_experts
        self.zero_expert_num = zero_expert_num
        self.zero_expert_type = zero_expert_type

        # Expert mapping used in self.load_weights
        self.expert_mapping = expert_mapping

        # Round up hidden size if needed.
        hidden_size = maybe_roundup_hidden_size(
            hidden_size, moe_in_dtype, quant_config, self.moe_parallel_config
        )

        # For smuggling this layer into the fused moe custom op
        compilation_config = vllm_config.compilation_config
        
        if hasattr(compilation_config, "static_forward_context"):
            if prefix in compilation_config.static_forward_context:
                raise ValueError("Duplicate layer name: {}".format(prefix))
            compilation_config.static_forward_context[prefix] = self
            
        self.layer_name = prefix

        self.enable_eplb = enable_eplb
        self.expert_load_view: torch.Tensor | None = None
        self.logical_to_physical_map: torch.Tensor | None = None
        self.logical_replica_count: torch.Tensor | None = None

        # ROCm aiter shared experts fusion
        self.num_fused_shared_experts = (
            n_shared_experts
            if n_shared_experts is not None
            and is_rocm_aiter_fusion_shared_expert_enabled()
            else 0
        )
        if (
            not is_rocm_aiter_fusion_shared_expert_enabled()
            and self.num_fused_shared_experts != 0
        ):
            raise ValueError(
                "n_shared_experts is only supported on ROCm aiter when "
                "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS is enabled"
            )

        # Determine expert maps
        if self.use_ep:
            if self.enable_eplb:
                assert self.global_num_experts % self.ep_size == 0, (
                    "EPLB currently only supports even distribution of "
                    "experts across ranks."
                )
            else:
                assert num_redundant_experts == 0, (
                    "Redundant experts are only supported with EPLB."
                )

            expert_placement_strategy = (
                vllm_config.parallel_config.expert_placement_strategy
            )
            if expert_placement_strategy == "round_robin":
                # TODO(Bruce): will support round robin expert placement with
                # EPLB enabled in the future.
                round_robin_supported = (
                    (num_expert_group is not None and num_expert_group > 1)
                    and num_redundant_experts == 0
                    and not self.enable_eplb
                )

                if not round_robin_supported:
                    logger.warning(
                        "Round-robin expert placement is only supported for "
                        "models with multiple expert groups and no redundant "
                        "experts. Falling back to linear expert placement."
                    )
                    expert_placement_strategy = "linear"

            self.expert_map: torch.Tensor | None
            local_num_experts, expert_map, expert_mask = determine_expert_map(
                ep_size=self.ep_size,
                ep_rank=self.ep_rank,
                global_num_experts=self.global_num_experts,
                expert_placement_strategy=expert_placement_strategy,
                num_fused_shared_experts=self.num_fused_shared_experts,
            )
            self.local_num_experts = local_num_experts
            self.register_buffer("expert_map", expert_map)
            self.register_buffer("expert_mask", expert_mask)
            logger.info_once(
                "[EP Rank %s/%s] Expert parallelism is enabled. Expert "
                "placement strategy: %s. Local/global"
                " number of experts: %s/%s. Experts local to global index map:"
                " %s.",
                self.ep_rank,
                self.ep_size,
                expert_placement_strategy,
                self.local_num_experts,
                self.global_num_experts,
                get_compressed_expert_map(self.expert_map),
            )
        else:
            self.local_num_experts, self.expert_map, self.expert_mask = (
                self.global_num_experts,
                None,
                None,
            )

        self.top_k = top_k

        self._init_aiter_shared_experts_topK_buffer(
            vllm_config=vllm_config, dp_size=dp_size_
        )

        assert intermediate_size % self.tp_size == 0
        self.hidden_size = hidden_size
        self.intermediate_size_per_partition = intermediate_size // self.tp_size
        self.reduce_results = reduce_results
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        if self.use_grouped_topk:
            assert num_expert_group is not None and topk_group is not None
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.custom_routing_function = custom_routing_function
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self.e_score_correction_bias = e_score_correction_bias
        self.apply_router_weight_on_input = apply_router_weight_on_input
        self.activation = activation

        if self.scoring_func != "softmax" and not self.use_grouped_topk:
            raise ValueError(
                "Only softmax scoring function is supported for non-grouped topk."
            )

        moe = MorehFusedMoEConfig(
            num_experts=self.global_num_experts,
            experts_per_token=top_k,
            hidden_dim=hidden_size,
            num_local_experts=self.local_num_experts,
            moe_parallel_config=self.moe_parallel_config,
            in_dtype=moe_in_dtype,
            max_num_tokens=getattr(envs, 'VLLM_MOE_DP_CHUNK_SIZE', 256),
            has_bias=has_bias,
        )
        self.moe_config = moe
        self.moe_quant_config: FusedMoEQuantConfig | None = None
        self.quant_config = quant_config

        # Note: get_quant_method will look at the layer's local_num_experts
        # for heuristic purposes, so it must be initialized first.
        quant_method: QuantizeMethodBase | None = None
        # quant_method = (
        #     MorehUnquantizedFusedMoEMethod(moe)
        #     if quant_config is None
        #     else quant_config.get_quant_method(self, prefix)
        # )
        # if quant_method is None:
        #     quant_method = MorehUnquantizedFusedMoEMethod(moe)
        quant_method = quant_config.get_quant_method(self, prefix)
        
        assert quant_method is not None
        assert isinstance(quant_method, FusedMoEMethodBase)
        self.quant_method = quant_method

        if self.enable_eplb:
            from vllm.model_executor.layers.quantization.fp8 import Fp8MoEMethod

            if not isinstance(quant_method, (Fp8MoEMethod)):
                # TODO: Add support for additional quantization methods.
                # The implementation for other quantization methods does not
                # contain essential differences, but the current quant API
                # design causes duplicated work when extending to new
                # quantization methods, so I'm leaving it for now.
                # If you plan to add support for more quantization methods,
                # please refer to the implementation in `Fp8MoEMethod`.
                raise NotImplementedError(
                    "EPLB is only supported for FP8 quantization for now."
                )

        moe_quant_params = {
            "num_experts": self.local_num_experts,
            "hidden_size": hidden_size,
            "intermediate_size_per_partition": self.intermediate_size_per_partition,
            "params_dtype": params_dtype,
            "weight_loader": self.weight_loader,
        }
        # need full intermediate size pre-sharding for WNA16 act order
        if self.quant_method.__class__.__name__ in (
            "GPTQMarlinMoEMethod",
            "CompressedTensorsWNA16MarlinMoEMethod",
            "CompressedTensorsWNA16MoEMethod",
        ):
            moe_quant_params["intermediate_size_full"] = intermediate_size
            
        if self.quant_method.__class__.__name__ == "MorehMxfp4MoEMethod":
            # print(f"Using MorehMxfp4MoEMethod")
            moe_quant_params["intermediate_size_full"] = intermediate_size
            moe_quant_params["tp_size"] = self.tp_size
            moe_quant_params["intermediate_size_per_partition_before_pad"] = self.intermediate_size_per_partition

        self.quant_method.create_weights(layer=self, **moe_quant_params)

        # Chunked all2all staging tensor
        self.batched_hidden_states: torch.Tensor | None = None
        self.batched_router_logits: torch.Tensor | None = None

        if self.use_dp_chunking:
            states_shape: tuple[int, ...]
            logits_shape: tuple[int, ...]

            # Note here we use `num_experts` which is logical expert count
            if vllm_config.parallel_config.enable_dbo:
                states_shape = (2, moe.max_num_tokens, self.hidden_size)
                logits_shape = (2, moe.max_num_tokens, num_experts)
            else:
                states_shape = (moe.max_num_tokens, self.hidden_size)
                logits_shape = (moe.max_num_tokens, num_experts)

            self.batched_hidden_states = torch.zeros(
                states_shape, dtype=moe.in_dtype, device=torch.cuda.current_device()
            )

            self.batched_router_logits = torch.zeros(
                logits_shape, dtype=moe.in_dtype, device=torch.cuda.current_device()
            )

    @property
    def use_mori_kernels(self):
        return self.moe_parallel_config.use_mori_kernels
    
    @property
    def use_dp_chunking(self) -> bool:
        # Route to the chunked forward path using the FlashInfer Cutlass kernel
        # only when data parallelism (DP) is enabled.
        return (
            self.moe_parallel_config.use_pplx_kernels
            or self.moe_parallel_config.use_deepep_ll_kernels
            or self.moe_parallel_config.use_mori_kernels
            or (self.dp_size > 1 and self.use_flashinfer_cutlass_kernels)
        )
    