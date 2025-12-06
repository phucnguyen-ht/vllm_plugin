import os 
from typing import Any, Callable

environment_variables: dict[str, Callable[[], Any]] = {
    # use attention backends from vllm_moreh
    "VLLM_MOREH_USE_CUSTOM_ATTN_BACKEND":
    lambda: (os.getenv("VLLM_MOREH_USE_CUSTOM_ATTN_BACKEND", "1").lower() in
             ("true", "1")),
    "VLLM_MOREH_USE_DUAL_MOE":
    lambda: (os.getenv("VLLM_MOREH_USE_DUAL_MOE", "1").lower() in
             ("true", "1")),
    "VLLM_MOREH_MORI_CONFIG_PATH": lambda: os.getenv("VLLM_MORI_CONFIG_PATH", None),       
    "VLLM_MOREH_ALL2ALL_BACKEND": lambda: os.getenv("VLLM_MOREH_ALL2ALL_BACKEND", "allgather_reducescatter"), 
}

def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return list(environment_variables.keys())