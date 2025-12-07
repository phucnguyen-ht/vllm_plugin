# TODO(anhduong): Refactor this file !!!
# SPDX-License-Identifier: Apache-2.0
import functools
import json
import os
from typing import List, Optional, Tuple

import torch

# from aiter.fused_moe import moe_sorting

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.platforms import current_platform

try:
    from vllm_plugin.quantization.utils.fused_moe_tuning import TuningConfig
except Exception as e:
    TuningConfig = object
    
# Import moreh_rocm_kernels
# import moreh_rocm_kernels._rocm_moreh_C

BLOCK_SIZE_M = 32
GPTOSS_LOCAL_BLOCK_SIZE_M = 16
# INTER_DEVICE_TP = 4  # NOTE(anhduong): Default inter-device TP for 1stage kernel (best performance)

DEVICE_NAME = torch.cuda.get_device_name().lower()

# # NOTE(anhduong): This is a tuned hyperparameter for best performance on MI300 and MI308x
# # TODO(anhduong): Write a script to automatically tune this hyperparameter
# gptoss_tune_path = None
# if 'mi300' in DEVICE_NAME:
#     MIN_TOKEN_2STAGES = 9
#     MAX_TOKEN_2STAGES = 9
#     gptoss_tune_path = f'configs/moe_2stages_gptoss_mi300x.csv'
# elif 'mi308x' in DEVICE_NAME:
#     MIN_TOKEN_2STAGES = 5
#     MAX_TOKEN_2STAGES = 5
#     gptoss_tune_path = f'configs/moe_2stages_gptoss_mi308x.csv'
# elif 'mi250' in DEVICE_NAME:
#     MIN_TOKEN_2STAGES = 512 # TODO(anhduong): Change name to "max_token" instead, because this is the max number of tokens that can be processed by 2stages kernel
#     MAX_TOKEN_2STAGES = 512
# else:
#     print(f"{DEVICE_NAME = } shouldn't be supported!")
#     MIN_TOKEN_2STAGES = 0
#     MAX_TOKEN_2STAGES = 0

# # TODO: Fix torch.cuda.get_device_name(), on rocm7, it return empty string
# # Temporarily set gptoss_tune_path to mi300x config
# gptoss_tune_path = f'configs/moe_2stages_gptoss_mi300x.csv'

logger = init_logger(__name__)

# ==================================================
# GPTOSS
# =================================================

ACTIVATION_NAME_TO_ID = {
    "swigluoai": 0,
    "silu": 1,
    "gelu": 2,
}

GELU_APPROX_NAME_TO_ID = {
    "none": 0,
    "tanh": 1,
}

import custom_moe_gfx928

# Original implementation: https://github.com/ROCm/aiter/blob/main/op_tests/test_moe_sorting.py 
def moe_sorting_naive(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    expert_mask=None,
    block_size:int=BLOCK_SIZE_M
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    device = topk_ids.device
    M, topk = topk_ids.shape
    topk = topk_ids.shape[1]
    max_num_tokens_padded = topk_ids.numel() + num_experts * block_size - topk
    max_num_m_blocks = int((max_num_tokens_padded + block_size - 1) // block_size)
    init_val = topk << 24 | M
    sorted_ids = torch.full(
        (max_num_tokens_padded,), init_val, dtype=torch.int32, device=device
    )
    sorted_weights = torch.empty(
        (max_num_tokens_padded,), dtype=torch.float, device=device
    )
    sorted_expert_ids = torch.full(
        (max_num_m_blocks,), -1, dtype=torch.int32, device=device
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=device)
    
    sorted_ids_begin = 0
    sorted_expert_ids_begin = 0
    skip_expert_num = 0
    for expertId in range(num_experts):
        if expert_mask != None and expert_mask[expertId] == 0:
            skip_expert_num += 1
            continue
        token_id, topk_id = torch.where(topk_ids == expertId)
        tokensNum = token_id.numel()
        sorted_expert_ids_num = (tokensNum + block_size - 1) // block_size
        tokensNumPad = sorted_expert_ids_num * block_size
        sorted_ids[sorted_ids_begin : sorted_ids_begin + tokensNum] = (
            topk_id << 24 | token_id
        )
        sorted_weights[sorted_ids_begin : sorted_ids_begin + tokensNum] = topk_weights[
            token_id, topk_id
        ]
        sorted_ids_begin = sorted_ids_begin + tokensNumPad
        sorted_expert_ids[
            sorted_expert_ids_begin : sorted_expert_ids_begin + sorted_expert_ids_num
        ] = (expertId - skip_expert_num)
        sorted_expert_ids_begin = sorted_expert_ids_begin + sorted_expert_ids_num

    num_tokens_post_pad[0] = sorted_ids_begin

    return sorted_ids, sorted_weights, sorted_expert_ids, num_tokens_post_pad

def rocm_gptoss_moreh_moe_1stage(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    # use_fp8_w8a8: bool = False,
    # use_int8_w8a8: bool = False,
    # use_int8_w8a16: bool = False,
    # use_int4_w4a16: bool = False,
    global_num_experts: int = -1,
    # per_channel_quant: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
    # w1_zp: Optional[torch.Tensor] = None,
    # w2_zp: Optional[torch.Tensor] = None,
    a_scale: Optional[torch.Tensor] = None,
    # a1_scale: Optional[torch.Tensor] = None,
    # a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    expert_map: Optional[torch.Tensor] = None,
    tuning_config: Optional[TuningConfig] = None,
    # shuffle_in: int = 16,
    # shuffle_ik: int = 16
) -> torch.Tensor:
    
    # logger.info_once(f"[Custom Moreh MoE] Using GPTOSS Moreh MoE 1stage with token={hidden_states.shape[0]}, E={w1.shape[0]}, topk={topk_ids.shape[1]}, model_dim={hidden_states.shape[1]}, inter_dim={w2.shape[1]}")
    # assert a1_scale is None # NOTE(anhduong): Only support a1_scale is None to force use blockscale quantization
    # a_q, a_scale = aiter_quant_fp8_blockscale(hidden_states, transpose_scale=False)

    tuning_config = tuning_config or TuningConfig()
    num_token = hidden_states.shape[0]
    model_dim = hidden_states.shape[-1]
    inter_dim = w2.shape[-1]
    topk = topk_ids.shape[-1]
    E = w1.shape[0]
    LOCAL_BLOCK_SIZE_M = tuning_config.LOCAL_BLOCK_SIZE_M or 16
    alpha = 1.702
    limit = 7.0
    
    sorted_token_ids, sorted_weight_buf, sorted_expert_ids, num_valid_ids = \
        moe_sorting_naive(topk_ids, topk_weights, E, None, LOCAL_BLOCK_SIZE_M)
        
    out_ck = torch.empty((num_token, model_dim), dtype=hidden_states.dtype, device=hidden_states.device)

    # gelu_approx = "none"
    # activation_id = ACTIVATION_NAME_TO_ID[activation]
    # gelu_mode = GELU_APPROX_NAME_TO_ID.get(gelu_approx, 0)
    # SHUFFLE_IN = shuffle_in
    # SHUFFLE_IK = shuffle_ik
    # num_parallel_by_interdim = tuning_config.PARALLEL_INTER_DIM or 1

    gptoss_moe_1stage(
        hidden_states, w1, w2, None, w1_scale, w2_scale,
        w1_bias, w2_bias,
        out_ck, sorted_token_ids,
        sorted_weight_buf, sorted_expert_ids,
        num_valid_ids, LOCAL_BLOCK_SIZE_M, topk,
        inter_dim, model_dim, alpha, limit
    )
    
    return out_ck

def gptoss_moe_1stage(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    a_scale: Optional[torch.Tensor],
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w1_bias: torch.Tensor,
    w2_bias: torch.Tensor,
    output: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_weight: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    tile_size: int,
    topk: int,
    inter_dim: int,
    dim: int,
    alpha: float,
    limit: float,
    # activation_id: int,
    # gelu_mode: int,
    # shuffle_in: int,
    # shuffle_ik: int,
    # num_parallel_by_interdim: int,
    # use_ntload_stage1: bool = False,
    # use_ntload_stage2: bool = False,
) -> torch.Tensor:
    if not current_platform.is_rocm():
        raise NotImplementedError(
            "The optimized launch_GptossMoe2Stages kernel is only "
            "available on ROCM platform. Current device: "
            f"{torch.cuda.get_device_name().lower()}")
    
    custom_moe_gfx928.launch_FusedMoeWmxfp4A8(
        hidden_states, w1, w2, a_scale, w1_scale, w2_scale,
        w1_bias, w2_bias,
        output, sorted_token_ids,
        sorted_weight, sorted_expert_ids,
        num_valid_ids, tile_size, topk,
        inter_dim, dim, alpha, limit,
        # activation_id, gelu_mode,
        # shuffle_in, shuffle_ik, num_parallel_by_interdim,
        # use_ntload_stage1, use_ntload_stage2,
    )

    return output