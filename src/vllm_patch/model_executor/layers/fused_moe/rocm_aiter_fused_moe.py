from functools import cache

from vllm.platforms import current_platform
from vllm import envs

@cache
def is_rocm_aiter_moe_enabled() -> bool:
    if hasattr(envs, 'VLLM_ROCM_USE_AITER_MOE'):
        return (
            current_platform.is_rocm()
            and envs.VLLM_ROCM_USE_AITER_MOE
            and envs.VLLM_ROCM_USE_AITER
        )
    else:
        return False

@cache
def is_rocm_aiter_fusion_shared_expert_enabled() -> bool:
    if hasattr(envs, 'VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS'):
        return (
            envs.VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS and is_rocm_aiter_moe_enabled()
        )
    else:
        return False