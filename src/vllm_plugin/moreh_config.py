import json
import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional

from vllm import envs

this_dir = os.path.dirname(os.path.abspath(__file__))
VLLM_MOREH_ROOT_DIR = os.path.abspath(f"{this_dir}")

MOREH_QUANTIZATION_METHOD = ["mixed", "dual", "fp8_per_block_to_per_token", "moreh_mxfp4", "mxfp4"]

@dataclass
class MorehCompilationConfig:
    max_batch_size_to_capture: int = None
    vllm_compilation_config: dataclass = None  # Need this to force set cudagraph_capture_sizes

    def __post_init__(self):
        # We use __post_init__ here since we need to set cudagraph_capture_sizes
        # after the object is initialized.
        use_v1 = getattr(envs, "VLLM_USE_V1", False)
        capture_step = getattr(envs, "VLLM_V1_CUDAGRAPH_CAPTURE_STEP", None)
        if (use_v1 and capture_step and capture_step > 0
                and self.max_batch_size_to_capture is not None):
            cudagraph_capture_sizes = list(
                range(1, self.max_batch_size_to_capture + 1,
                      capture_step))
            self.vllm_compilation_config.cudagraph_capture_sizes = cudagraph_capture_sizes
            self.vllm_compilation_config.init_with_cudagraph_sizes(
                cudagraph_capture_sizes)


@dataclass
class MorehConfig:
    compilation_config: MorehCompilationConfig

    @classmethod
    def init_moreh_config(cls, vllm_config: dataclass) -> 'MorehConfig':
        additional_config = vllm_config.additional_config
        compilation_config = additional_config.get("compilation_config", {})
        return cls(compilation_config=MorehCompilationConfig(
            vllm_compilation_config=vllm_config.compilation_config,
            **compilation_config))


### Customized post-parsed functions for Moreh ###
"""
These functions should be triggered directly after post-parsing the arguments in VLLM.

For example, the `_verify_quantization` function is used to verify the quantization configuration
for Moreh, which is different from the default VLLM behavior. This is because Moreh supports mixed 
quantization, which is not supported by VLLM. If we don't override this function, VLLM will raise 
an error when it detects a mixed quantization configuration:
'''
elif self.quantization != quant_method:
    raise ValueError(
        "Quantization method specified in the model config "
        f"({quant_method}) does not match the quantization "
        f"method specified in the `quantization` argument "
        f"({self.quantization}).")
'''
"""


def _verify_quantization(self, original_func: Optional[callable] = None):
    # Mixed quantization is supported for Moreh, so we don't need to verify it
    if self.quantization not in MOREH_QUANTIZATION_METHOD:
        original_func(self)
    return


def _get_quantization_config(
        model_config: dataclass,
        load_config: dataclass,
        original_func: Optional[callable] = None) -> Optional[Any]:
    # Mixed quantization config
    if model_config.quantization == "mixed":
        mixed_quantization_config_path = model_config.kwargs.get(
            "mixed_quantization_filepath",
            os.path.join(model_config.model, "mixed_quantization_config.json"))
        with open(mixed_quantization_config_path) as f:
            mixed_quantization_config_dict = json.load(f)
        from vllm_moreh.quantization.mixed import MixedConfig
        return MixedConfig.from_config(mixed_quantization_config_dict)
    else:
        return original_func(model_config, load_config)


# Use for setting up a custom validation function for class methods
# While for static method, we set it up directly
def setup_custom_validation_function(cls: type, func_name: str,
                                     custom_func: callable):
    assert "original_func" in custom_func.__code__.co_varnames, "custom_func must have 'original_func' as a parameter"

    original_func = getattr(cls, func_name)

    def wrapper(self, *args, **kwargs):
        return custom_func(self, *args, original_func=original_func, **kwargs)

    setattr(cls, func_name, wrapper)


# Register custom post-parsed functions for Moreh
def register_post_parsed_function():
    from vllm.config import ModelConfig, VllmConfig
    setup_custom_validation_function(ModelConfig, "_verify_quantization",
                                     _verify_quantization)
    VllmConfig._get_quantization_config = partial(
        _get_quantization_config,
        original_func=VllmConfig._get_quantization_config)


### Current Moreh config ###
_current_moreh_config: Optional[MorehConfig] = None


def set_current_moreh_config(vllm_config: dataclass) -> None:
    global _current_moreh_config
    _current_moreh_config = MorehConfig.init_moreh_config(vllm_config)


def get_current_moreh_config() -> MorehConfig:
    global _current_moreh_config
    assert _current_moreh_config is not None, \
        "Current Moreh config is not set. " \
        "Please call set_current_moreh_config before using this function."
    return _current_moreh_config
