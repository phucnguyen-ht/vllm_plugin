from vllm.platforms.rocm import RocmPlatform
from vllm.logger import init_logger
from vllm.platforms.interface import _Backend

from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.platforms.rocm import RocmPlatform
from vllm.platforms.interface import _Backend

from vllm_plugin import envs as moreh_envs
from vllm_plugin.moreh_config import (MOREH_QUANTIZATION_METHOD,
                                     set_current_moreh_config)

from .utils import on_mi250x

logger = init_logger(__name__)


class MorehRocmPlatform(RocmPlatform):
    device_control_env_var: str = "HIP_VISIBLE_DEVICES"
    supported_quantization = [
        *RocmPlatform.supported_quantization, *MOREH_QUANTIZATION_METHOD
    ]

    # TODO: implement dispatch flags: is_rocm_aiter_*
    @classmethod
    def pre_register_and_update(cls, parser=None):
        # if parser is not None:
        #     quant_action = parser._option_string_actions.get("--quantization")
        #     if quant_action and hasattr(quant_action, "choices"):
        #         if "mixed" not in quant_action.choices:
        #             quant_action.choices.append("mixed")
        pass

    @classmethod
    def check_and_update_config(cls, vllm_config: dataclass) -> None:
        set_current_moreh_config(vllm_config)

        # [Moreh] Override all2all backend from Moreh env var
        parallel_config = vllm_config.parallel_config
        
        if hasattr(parallel_config, "all2all_backend"):
            parallel_config.all2all_backend = moreh_envs.VLLM_MOREH_ALL2ALL_BACKEND
        else:
            print(f"[WARN] {parallel_config.__class__.name} doesn't have 'all2all_backend' property!")
        
        super().check_and_update_config(vllm_config)

    @classmethod
    def get_attn_backend_cls(cls, *args, **kwargs) -> str: 
        # TODO: add moreh custom backend selection logic here
        return super().get_attn_backend_cls(*args, **kwargs)
    
    # @classmethod
    # def get_device_communicator_cls(cls) -> str:
    #     return (
    #         "vllm_plugin.distributed.device_communicators.cuda_communicator.MorehCudaCommunicator"  # noqa
    #     )